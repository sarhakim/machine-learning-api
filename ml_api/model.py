import logging
import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from ml_api.config import Columns, Paths, Params


class Model:
    """Class to build the model from a provided dataset."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
        self.precision = None
        self.recall = None
        self.result_matrix = None

    def build(self):
        """ Build the model."""

        logging.info(f"Processing input data.")
        processed_df = self.process_categorical_data(self.dataset)
        test, train = self._split_train_test(processed_df)

        x_train, y_train = self._extract_features_and_target(train)
        x_test, y_test = self._extract_features_and_target(test)

        logging.info(f"Building model.")
        model = RandomForestClassifier()
        model.fit(X=x_train, y=y_train)

        self.model = model

        self._model_performances(x_test, y_test)

        if self.precision > Params.precision_threshold:
            logging.info("Saving the model.")
            self._save_model(processed_df_columns=processed_df.columns)
        else:
            logging.info(f"Model precision is not above threshold. "
                         f"The model will not be saved.")

    @staticmethod
    def process_categorical_data(data_df):
        """
        Creates dummy boolean columns from categorical data.
        Args:
            data_df: pandas dataframe

        Returns: Pandas dataframe with new columns
        """
        return pd.get_dummies(data_df, columns=Columns.categorical)

    @staticmethod
    def _split_train_test(processed_df):
        train = processed_df.query(f'{Columns.set} == "TRAIN"').drop('set', axis=1)
        test = processed_df.query(f'{Columns.set} == "TEST"').drop('set', axis=1)
        return test, train

    @staticmethod
    def _extract_features_and_target(df):
        x = df.drop(Columns.label, axis=1)
        y = df[Columns.label]
        return x, y

    def _model_performances(self, x_test, y_test):
        predictions = self.model.predict(x_test)

        self.result_matrix = confusion_matrix(y_pred=predictions, y_true=y_test)
        self.precision = precision_score(y_pred=predictions, y_true=y_test)
        self.recall = recall_score(y_pred=predictions, y_true=y_test)

        logging.info(f"Precision : {self.precision}")
        logging.info(f"Recall : {self.precision}")
        logging.info(f"tn, fp, fn, tp : {self.result_matrix.ravel()}")

    def _save_model(self, processed_df_columns):
        current_ts = round(time.time())
        model_filename = f"model_{current_ts}.pkl"
        model_columns_filename = f"model_columns_{current_ts}.pkl"

        joblib.dump(self.model, os.path.join(Paths.model_dir, model_filename))
        model_columns = list(processed_df_columns)
        joblib.dump(model_columns, os.path.join(Paths.model_dir, model_columns_filename))
        logging.info(f"{model_filename} and {model_columns_filename} have been created.")
