import json

import pandas as pd
from sqlalchemy import Integer, String
from sqlalchemy import Table, Column, Float, MetaData
from sqlalchemy import create_engine

from ml_api.config import Paths


class Database:
    """Utils to call the postgressql database."""

    def __init__(self):
        self.engine = create_engine(Paths.database_uri)
        self.meta = MetaData(self.engine)
        self.table = Table(Paths.table,
                           self.meta,
                           Column('CONT1', Float),
                           Column('CONT2', Float),
                           Column('CONT3', Float),
                           Column('CONT4', Float),
                           Column('CONT5', Float),
                           Column('CONT6', Float),
                           Column('CAT1', Float),
                           Column('CAT2', Float),
                           Column('DIS1', Float),
                           Column('DIS2', Float),
                           Column('LABEL', Integer),
                           Column('set', String)
                           )

    def insert_data_in_db(self, json_values):
        with self.engine.connect() as conn:
            for json_value in json_values:
                insert_statement = self.table.insert().values(
                    CONT1=json_value.get('CONT1'),
                    CONT2=json_value.get('CONT2'),
                    CONT3=json_value.get('CONT3'),
                    CONT4=json_value.get('CONT4'),
                    CONT5=json_value.get('CONT5'),
                    CONT6=json_value.get('CONT6'),
                    CAT1=json_value.get('CAT1'),
                    CAT2=json_value.get('CAT2'),
                    DIS1=json_value.get('DIS1'),
                    DIS2=json_value.get('DIS2'),
                    LABEL=json_value.get('LABEL'),
                    set=json_value.get('set')
                )
                conn.execute(insert_statement)

    def create_table(self):
        with engine.connect():
            self.table.create()

    def get_data(self):
        with self.engine.connect() as conn:
            result_set = conn.execute(f"SELECT * FROM {Paths.table}")
        df = pd.DataFrame(result_set)
        df.columns = result_set.keys()
        return df

    def delete_data(self):
        with self.engine.connect() as conn:
            conn.execute(f"DELETE FROM {Paths.table}")

    def push_data_from_csv(self):
        df = pd.read_csv(Paths.data, sep=';', index_col=0)
        json_values = json.loads(df.to_json(orient='records'))
        self.insert_data_in_db(json_values=json_values)


if __name__ == '__main__':
    engine = create_engine(Paths.database_uri)
    Database().create_table()
