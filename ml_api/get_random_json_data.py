import pandas as pd


def main():
    df = pd.read_csv('../data/BINNING_data_set.csv', sep=';', index_col=0)
    train = df.query('set == "TRAIN"').drop('set', axis=1)
    test = df.query('set == "TEST"').drop('set', axis=1)

    X_train = train.drop('LABEL', axis=1)
    y_train = train['LABEL']

    X_test = test.drop('LABEL', axis=1)
    y_test = test['LABEL']

    print(X_test.iloc[:2, :].to_json(orient='records'))


if __name__ == '__main__':
    main()
