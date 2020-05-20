from statistics import mean
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

df_mat = pd.read_csv('Student_Mat.txt')
df_por = pd.read_csv('Student_Por.txt')
df_por._convert(numeric=True)
df_mat._convert(numeric=True)


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

drop_columns_mat = df_mat.drop(['G3'], 1).columns.values
drop_columns_por = df_por.drop(['G3'], 1).columns.values

for column in drop_columns_mat:

    X = np.array(df_mat.drop([column,'G3'], 1))
    X = preprocessing.scale(X)
    y = np.array(df_mat['G3'])

    accuracies = []
    for i in range(25):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.1)

        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        accuracies.append(accuracy)

    print("The Prediction Accuracy of Math score after dropping '"+ column +"' column is: " + str(mean(accuracies)))

print("\n")

for column in drop_columns_por:

    X = np.array(df_por.drop([column,'G3'], 1))
    X = preprocessing.scale(X)
    y = np.array(df_por['G3'])

    accuracies = []
    for i in range(25):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        accuracies.append(accuracy)

    print("The Prediction Accuracy of Portugese score after dropping '" + column + "' column is: " + str(mean(accuracies)))





