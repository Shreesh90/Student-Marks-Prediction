from statistics import mean
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")
from sklearn.utils import shuffle

df1 = pd.read_csv('Student_Mat.txt')
df2 = pd.read_csv('Student_Por.txt', sep=';')
df2._convert(numeric=True)
df1._convert(numeric=True)

df = pd.merge(df1, df2, on = [ "school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet" ])
df._convert(numeric=True)

df = df.rename(columns={"G1_x":"G1_Mat", "G2_x":"G2_Mat", "G3_x":"G3_Mat","G1_y":"G1_Por", "G2_y":"G2_Por", "G3_y":"G3_Por"})

df_mat = df[["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet","G1_Mat", "G2_Mat", "G3_Mat"]]
df_por = df[["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet","G1_Por", "G2_Por", "G3_Por"]]


def handle_non_numeric_data(df):
    columns = df.columns.values
    for column in columns:
        text_to_digit_value = {}

        def convert_to_digit(val):
            return text_to_digit_value[val]

        if df[column].dtype != np.int64 or df[column].dtype != np.float64:
            column_content = df[column].values.tolist()
            unique_elements = set(column_content)
            x = 1
            for element in unique_elements:
                if element not in text_to_digit_value:
                    text_to_digit_value[element] = x
                    x += 1

            df[column] = list(map(convert_to_digit, df[column]))

    return df

df_mat = handle_non_numeric_data(df_mat)
df_por = handle_non_numeric_data(df_por)

# df_mat_GP = df_mat[df["school"] == "GP"]
# df_mat_MS = df_mat[df["school"] == "MS"]
# df_por_GP = df_por[df["school"] == "GP"]
# df_por_MS = df_por[df["school"] == "MS"]
#
# mean_mat_GP = df_mat_GP["G3_Mat"].mean()
# mean_mat_MS = df_mat_MS["G3_Mat"].mean()
# mean_por_MS = df_por_MS["G3_Por"].mean()
# mean_por_GP = df_por_GP["G3_Por"].mean()
# print(mean_mat_GP)
# plt.bar(["GP_mat", "MS_mat", "GP_por", "MS_por"], [mean_mat_GP, mean_mat_MS, mean_por_GP, mean_por_MS])
# plt.show()


for i in range(25):
    df_mat = shuffle(df_mat)
    X_mat = np.array(df_mat.drop(['G3_Mat'], 1))
    X_mat = preprocessing.scale(X_mat)
    y_mat = np.array(df_mat['G3_Mat'])
    accuracies_mat = []
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_mat,y_mat,test_size = 0.1)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    accuracies_mat.append(accuracy)

print("Prediction accuracy of Math Scores in G3 for students is: " + str(mean(accuracies_mat)))


for i in range(25):
    df_por = shuffle(df_por)
    X_por = np.array(df_por.drop(['G3_Por'], 1))
    X_por = preprocessing.scale(X_por)
    y_por = np.array(df_por['G3_Por'])
    accuracies_por = []
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_por,y_por,test_size = 0.1)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    accuracies_por.append(accuracy)

print("Prediction accuracy of Portuguese Scores in G3 for students is: " + str(mean(accuracies_por)))
