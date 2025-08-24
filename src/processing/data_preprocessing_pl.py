import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

abnormal_file = "../../../data/ptbdb_abnormal.csv"
normal_file = "../../../data/ptbdb_normal.csv"

def load_ptbdb_data():
    df1 = pd.read_csv(abnormal_file, header=None)
    df2 = pd.read_csv(normal_file, header=None)
    data = pd.concat([df1, df2], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

def add_synthetic_patient_id_ptbdb(segments_per_patient=20):
    data = load_ptbdb_data()
    n_rows = data.shape[0]
    n_patients = int(np.ceil(n_rows / segments_per_patient))
    patient_ids = np.repeat(np.arange(n_patients), segments_per_patient)[:n_rows]
    data_with_pid = data.copy()
    data_with_pid.insert(0, 'patient_id', patient_ids)
    return data_with_pid, patient_ids

def split_by_groupshuffle_and_scale(data, patient_col='patient_id', test_size=0.2, random_state=42):
    groups = data[patient_col].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(data, groups=groups))

    train_data = data.iloc[train_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    train_data_nopid = train_data.drop(columns=[patient_col])
    test_data_nopid = test_data.drop(columns=[patient_col])

    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data_nopid)
    test_data_scaled = scaler.transform(test_data_nopid)

    X_train = train_data_scaled.reshape(train_data_scaled.shape[0], train_data_scaled.shape[1], 1)
    X_test = test_data_scaled.reshape(test_data_scaled.shape[0], test_data_scaled.shape[1], 1)

    Y_train = train_data_nopid[train_data_nopid.columns[-1]].to_numpy()
    Y_test = test_data_nopid[test_data_nopid.columns[-1]].to_numpy()

    return X_train, X_test, Y_train, Y_test, scaler

def ptbdb_data_scaled_pl(scaler, data=None, patient_col='patient_id'):
    if data is None:
        data = load_ptbdb_data()
    if patient_col in data.columns:
        data_nopid = data.drop(columns=[patient_col])
    else:
        data_nopid = data
    data_scaled = scaler.transform(data_nopid)
    X_all = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)
    Y_all = data_nopid[data_nopid.columns[-1]].to_numpy()
    return X_all, Y_all


mitbih_train =  "../../../data/mitbih_train.csv"
mitbih_test =  "../../../data/mitbih_test.csv"

def load_mitbih_data_pl():
    train_data = pd.read_csv(mitbih_train, header=None)
    test_data = pd.read_csv(mitbih_test, header=None)
    return train_data, test_data

def add_synthetic_patient_id(data, segments_per_patient=20, start_patient_id=0):
    n_rows = data.shape[0]
    n_patients = int(np.ceil(n_rows / segments_per_patient))
    patient_ids = np.arange(start_patient_id, start_patient_id + n_patients)
    assigned_ids = np.repeat(patient_ids, segments_per_patient)[:n_rows]
    data_with_pid = data.copy()
    data_with_pid.insert(0, 'patient_id', assigned_ids)
    next_patient_id = start_patient_id + n_patients
    return data_with_pid, next_patient_id

def split_and_scale_mitbih_pl(train_data, test_data, patient_col='patient_id'):
    # Drop patient_id col before scaling
    train_data_nopid = train_data.drop(columns=[patient_col])
    test_data_nopid = test_data.drop(columns=[patient_col])

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data_nopid)
    test_scaled = scaler.transform(test_data_nopid)

    # Reshape for CNN/RNN/Autoencoder input
    X_train = train_scaled.reshape(train_scaled.shape[0], train_scaled.shape[1], 1)
    X_test = test_scaled.reshape(test_scaled.shape[0], test_scaled.shape[1], 1)

    # Last column is label
    Y_train = train_data_nopid[train_data_nopid.columns[-1]].to_numpy()
    Y_test = test_data_nopid[test_data_nopid.columns[-1]].to_numpy()

    return X_train, X_test, Y_train, Y_test, scaler

def mitbih_data_scaled_pl(scaler, data_with_pid, patient_col='patient_id'):
    data_nopid = data_with_pid.drop(columns=[patient_col])
    data_scaled = scaler.transform(data_nopid)
    X_all = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)
    Y_all = data_nopid[data_nopid.columns[-1]].to_numpy()
    return X_all, Y_all