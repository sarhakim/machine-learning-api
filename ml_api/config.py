import os


class Paths:
    project_dir = os.path.dirname(os.path.dirname(__file__))
    data = os.path.join(project_dir, 'data/BINNING_data_set.csv')
    model_dir = os.path.join(project_dir, 'model')
    model = os.path.join(project_dir, 'model/model.pkl')
    model_columns = os.path.join(project_dir, 'model/model_columns.pkl')

    database_uri = 'postgres+psycopg2://postgres:123@localhost:5432/ml_api_data'
    table = 'table_paylead'


class Columns:
    set = 'set'
    label = 'LABEL'
    categorical = ['CAT1', 'CAT2']
