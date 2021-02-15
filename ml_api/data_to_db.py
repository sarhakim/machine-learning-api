import json

import pandas as pd
from sqlalchemy import create_engine

from ml_api.config import Paths
from ml_api.database import insert_data_in_db


def main():
    df = pd.read_csv(Paths.data, sep=';', index_col=0)
    json_values = json.loads(df.to_json(orient='records'))
    engine = create_engine(Paths.database_uri)
    insert_data_in_db(engine=engine, json_values=json_values)


if __name__ == '__main__':
    main()
