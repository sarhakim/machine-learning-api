import argparse
import logging
import traceback

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from ml_api.config import Paths, Columns
from ml_api.database import Database
from ml_api.model import Model

app = Flask(__name__)
db_utils = Database()

logging.basicConfig(level=logging.INFO)


@app.route('/')
def hello():
    return 'Welcome to machine learning model APIs!'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query_df = pd.DataFrame(json_)
        processed = Model.process_categorical_data(query_df)

        model = joblib.load(Paths.model)
        logging.info('Model loaded')
        model_columns = joblib.load(Paths.model_columns)
        logging.info('Model loaded')

        reindexed = processed.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(reindexed)
        return jsonify({'prediction': list(prediction)})
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/insert_data', methods=['PUT'])
def insert_data():
    try:
        json_ = request.json
        db_utils.insert_data_in_db(json_values=json_)
        return jsonify({'Number of lines': len(json_)})
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/train_from_db', methods=['POST'])
def train_from_db():
    try:
        logging.info('train from db')
        df = db_utils.get_data()
        model_builder = Model(dataset=df)
        model_builder.build()
        return jsonify({
            'train labels number': len(df.query(f'{Columns.set} == "TRAIN"')),
            'precision': model_builder.precision,
            'recall': model_builder.recall,
            'tn, fp, fn, tp': str(model_builder.result_matrix.ravel())
        })
    except:
        return jsonify({'trace': traceback.format_exc()})


def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction API argument parser')
    parser.add_argument('--port', default=5000, type=int, help='port used to run the api')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    port = args.port
    app.run(port=port, debug=True)
