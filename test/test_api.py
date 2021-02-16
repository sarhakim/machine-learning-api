"""Class to test api.py module."""
import json
from unittest import TestCase
from unittest.mock import patch
import pandas as pd

from ml_api import api


def mock_insert_data(json_values):
    return None

def mock_save_model_data(processed_df_columns):
    return None


class TestApi(TestCase):
    """Class to test api module."""

    def setUp(self):
        self.app = api.app.test_client()
        self.app.testing = True
        self.dataset = [
        {"CONT1": 50.0, "CONT2": 6.0, "CONT3": 100.0, "CONT4": 50.0, "CONT5": 34.0998619795, "CONT6": 71.0,
         "CAT1": 11.0, "CAT2": 1.0, "DIS1": 1.0, "DIS2": 2.0, "set": "TRAIN", "LABEL": 1},
        {"CONT1": 100.0, "CONT2": 4.0, "CONT3": 100.0, "CONT4": 100.0, "CONT5": 34.8963916302, "CONT6": 100.0,
         "CAT1": 3.0, "CAT2": 1.0, "DIS1": 1.0, "DIS2": 6.0, "set": "TEST", "LABEL": 1}
        ]

    def test_hello(self):
        # When
        response = self.app.get('/', headers={"Content-Type": "application/json"})

        # Then
        expected_status = 200
        expected_response_data = 'Welcome to machine learning model APIs!'
        self.assertEqual(response.status_code, expected_status)
        self.assertEqual(str(response.get_data()), expected_response_data)

    @patch('ml_api.config.Paths.database_uri', "postgres+psycopg2://postgres:fake@fakehost:0/fake_db")
    @patch('ml_api.database.Database.insert_data_in_db')
    def test_insert_data(self, mock_insert_data_in_db):
        # Given
        mock_insert_data_in_db.side_effect = mock_insert_data(json_values='')

        # When
        response = self.app.put(
            '/insert_data', headers={"Content-Type": "application/json"}, data=json.dumps(self.dataset)
        )
        actual_status = response.status_code
        actual_output = json.loads(response.get_data())

        # Then
        expected_status = 200
        expected_response_data = {'Number of lines': 2}
        self.assertEqual(actual_status, expected_status)
        self.assertEqual(actual_output, expected_response_data)

    @patch('ml_api.config.Paths.database_uri', "postgres+psycopg2://postgres:fake@fakehost:0/fake_db")
    @patch('ml_api.model.Model._save_model')
    @patch('ml_api.database.Database.get_data')
    def test_train_from_db(self, mock_get_data, mock_save_model):
        # Given
        df = pd.DataFrame(self.dataset)
        mock_get_data.return_value = df
        mock_save_model.side_effect = mock_save_model_data(processed_df_columns='')

        # When
        response = self.app.post(
            '/train_from_db', headers={"Content-Type": "application/json"}, data=json.dumps(self.dataset)
        )
        actual_status = response.status_code
        actual_output = json.loads(response.get_data())
        actual_output_keys = set(actual_output.keys())

        # Then
        expected_status = 200
        expected_keys = {'train labels number', 'precision', 'recall', 'tn, fp, fn, tp'}
        expected_train_labels_number = 1

        self.assertEqual(actual_status, expected_status)
        self.assertEqual(actual_output_keys, expected_keys)
        self.assertEqual(actual_output.get('train labels number'), expected_train_labels_number)
