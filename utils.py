import pandas as pd


def remove_if_duplicates(train_data):
    duplicates_train = train_data.duplicated().sum()
    print('Duplicates in train data: {0}'.format(duplicates_train))

    train_data.drop_duplicates(keep='first', inplace=True)
    duplicates_train = train_data.duplicated().sum()

    print('Train data shape:', train_data.shape)
    print('Duplicates in train data: {0}'.format(duplicates_train))

    return train_data


def remove_if_duplicates_weights(train_data: pd.DataFrame):
    duplicates_train = train_data.duplicated().sum()
    print('Duplicates in train data: {0}'.format(duplicates_train))

    weights = train_data.value_counts().values

    train_data.drop_duplicates(keep='first', inplace=True)
    duplicates_train = train_data.duplicated().sum()

    print('Train data shape:', train_data.shape)
    print('Duplicates in train data: {0}'.format(duplicates_train))

    return train_data, weights

