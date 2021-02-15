import argparse
import traceback

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from sqlalchemy import create_engine

from ml_api.config import Paths
from ml_api.database import insert_data, get_train_data
from ml_api.model import Model

app = Flask(__name__)


@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query_df = pd.DataFrame(json_)
        processed = Model.process_categorical_data(query_df)
        reindexed = processed.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(reindexed)
        return jsonify({'prediction': list(prediction)})
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/insert_data', methods=['POST'])
def insert_train_data():
    try:
        json_ = request.json
        engine = create_engine(Paths.database_uri)
        insert_data(engine=engine, json_values=json_)
        return jsonify({'Number of lines': len(json_)})
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/train', methods=['POST'])
def train():
    try:
        json_ = request.json
        query_df = pd.DataFrame(json_)
        model_builder = Model(dataset=query_df)
        model_builder.build()
        return jsonify({'trace': "1"})
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/train_from_db', methods=['POST'])
def train_from_db():
    try:
        df = get_train_data()
        model_builder = Model(dataset=df)
        model_builder.build()
        return jsonify({'trace': "1"})
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

    model = joblib.load(Paths.model)  # Load "model.pkl"
    print('Model loaded')
    model_columns = joblib.load(Paths.model_columns)  # Load "model_columns.pkl"
    print('Model columns loaded')

    app.run(port=port, debug=True)
