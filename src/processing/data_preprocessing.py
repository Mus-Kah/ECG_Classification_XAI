import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

mitbih_train =  "../../data/mitbih_train.csv"
mitbih_test =  "../../data/mitbih_test.csv"

def load_mitbih_data(mitbih_train=mitbih_train, mitbih_test=mitbih_test):
    train_data = pd.read_csv(mitbih_train, header=None)
    test_data = pd.read_csv(mitbih_test, header=None)
    return train_data, test_data

def split_and_scale_mitbih_data(train_data, test_data):
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    X_train = train_data_scaled.reshape(train_data_scaled.shape[0], train_data_scaled.shape[1], 1)
    X_test = test_data_scaled.reshape(test_data_scaled.shape[0], test_data_scaled.shape[1], 1)

    Y_train = train_data[train_data.columns[-1]]
    Y_test = test_data[test_data.columns[-1]]

    return X_train, X_test, Y_train, Y_test, scaler

def mitbih_data_scaled(scaler):
    train_data, test_data = load_mitbih_data(mitbih_train, mitbih_test)
    data = pd.concat([train_data, test_data], ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)
    data_scaled = scaler.fit_transform(data)
    X_all = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)
    Y_all = data[data.columns[-1]].to_numpy()
    return X_all, Y_all

abnormal_file = "../../data/ptbdb_abnormal.csv"
normal_file = "../../data/ptbdb_normal.csv"

def load_ptbdb_data(abnormal_file=abnormal_file, normal_file=normal_file):
    df1 = pd.read_csv(abnormal_file, header=None)
    df2 = pd.read_csv(normal_file, header=None)
    data = pd.concat([df1, df2], ignore_index=True)
    return data.sample(frac=1).reset_index(drop=True)

def split_and_scale_ptbdb_data(data, test_size=0.27, random_state=42):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    X_train = train_data_scaled.reshape(train_data_scaled.shape[0], train_data_scaled.shape[1], 1)
    X_test = test_data_scaled.reshape(test_data_scaled.shape[0], test_data_scaled.shape[1], 1)

    Y_train = train_data[train_data.columns[-1]]
    Y_test = test_data[test_data.columns[-1]]

    return X_train, X_test, Y_train, Y_test, scaler

def ptbdb_data_scaled(scaler):
    data = load_ptbdb_data()
    data_scaled = scaler.fit_transform(data)
    X_all = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)
    Y_all = data[data.columns[-1]].to_numpy()
    return X_all, Y_all
