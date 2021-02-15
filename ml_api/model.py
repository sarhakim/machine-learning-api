import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from ml_api.config import Columns, Paths


class Model:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None

    def build(self):
        processed_df = self.process_categorical_data(self.dataset)
        test, train = self.split_train_test(processed_df)

        X_train, y_train = self.extract_features_target(train)
        X_test, y_test = self.extract_features_target(test)

        model = RandomForestClassifier()
        model.fit(X=X_train, y=y_train)

        self.model = model
        self.save_model(processed_df_columns=processed_df.columns)

    def save_model(self, processed_df_columns):
        current_ts = round(time.time())
        model_filename = f"model_{current_ts}.pkl"
        model_columns_filename = f"model_columns_{current_ts}.pkl"

        joblib.dump(self.model, os.path.join(Paths.model_dir, model_filename))
        model_columns = list(processed_df_columns)
        joblib.dump(model_columns, os.path.join(Paths.model_dir, model_columns_filename))

    def model_performance(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        result_matrix = confusion_matrix(y_pred=predictions, y_true=y_test)
        tn, fp, fn, tp = result_matrix.ravel()
        print(result_matrix)
        print(precision_score(y_pred=predictions, y_true=y_test))
        print(recall_score(y_pred=predictions, y_true=y_test))

    @staticmethod
    def extract_features_target(df):
        X = df.drop(Columns.label, axis=1)
        y = df[Columns.label]
        return X, y

    @staticmethod
    def split_train_test(processed_df):
        train = processed_df.query(f'{Columns.set} == "TRAIN"').drop('set', axis=1)
        test = processed_df.query(f'{Columns.set} == "TEST"').drop('set', axis=1)
        return test, train

    @staticmethod
    def process_categorical_data(data_df):
        return pd.get_dummies(data_df, columns=Columns.categorical)


if __name__ == '__main__':
    df = pd.read_csv(Paths.data, sep=';', index_col=0)
    Model().build()
