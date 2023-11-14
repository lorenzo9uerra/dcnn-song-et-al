import os
import numpy as np
from binascii import unhexlify
from bitstring import BitArray
import pandas as pd
import glob
import pathlib
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def pad_id(id_bits):
    while len(id_bits) != 29:
        id_bits.insert(0, 0)
    return id_bits

def build_frames(ids, labels, sequence_length=29):
    X, y = [], []
    ids = [BitArray(unhexlify(hex(x)[2:].upper().rjust(4, "0"))).bin for x in ids]
    num_samples = len(ids)-(len(ids)%sequence_length)
    print(num_samples)
    for i in range(0,num_samples,sequence_length):
        sequence = []
        for j in ids[i : i + sequence_length]:
            sequence.append(pad_id([int(x) for x in list(j)]))
        label = 0 if all(labels[i : i + sequence_length] == 0) else 1
        X.append(sequence)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

def load_data_road(attack_type):
    if (
        os.path.exists(f"data/preprocessed/road/{attack_type}_features_train.npy")
        and os.path.exists(f"data/preprocessed/road/{attack_type}_labels_train.npy")
        and os.path.exists(f"data/preprocessed/road/{attack_type}_features_test.npy")
        and os.path.exists(f"data/preprocessed/road/{attack_type}_labels_test.npy")
    ):
        print("Loading Preprocessed Data")
        X_train = np.load(f"data/preprocessed/road/{attack_type}_features_train.npy")
        y_train = np.load(f"data/preprocessed/road/{attack_type}_labels_train.npy")
        X_test = np.load(f"data/preprocessed/road/{attack_type}_features_test.npy")
        y_test = np.load(f"data/preprocessed/road/{attack_type}_labels_test.npy")
    else:
        # Path to the directory containing ROAD dataset CSV files
        directory_path = "data/road/"
        if "masquerade" in attack_type:
            print("_".join(attack_type.split("_")[:-1]))
            csv_files = glob.glob(
                directory_path
                + f'{"_".join(attack_type.split("_")[:-1])}_[1-3]_masquerade.csv'
            )
        else:
            csv_files = glob.glob(directory_path + f"{attack_type}_[1-3].csv")

        # Initialize an empty list to store individual DataFrames
        dataframes = []
        # dataframes.append(pd.read_csv("data/ambient.csv", names=["timestamp","ID","DATA0","DATA1","DATA2","DATA3","DATA4","DATA5","DATA6","DATA7","label"]))
        for file in csv_files:
            dataframe = pd.read_csv(
                file,
                names=[
                    "timestamp",
                    "ID",
                    "DATA0",
                    "DATA1",
                    "DATA2",
                    "DATA3",
                    "DATA4",
                    "DATA5",
                    "DATA6",
                    "DATA7",
                    "label",
                ],
            )
            dataframes.append(dataframe)
        if len(dataframes) > 1:
            data = pd.concat(dataframes, ignore_index=True)
        else:
            data = dataframes[0]
        data.columns = data.columns.str.strip()

        y = data["label"]
        X = data["ID"].apply(lambda x: int(x, 16))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        print(y_train.value_counts())
        smote = SMOTE(sampling_strategy={y_train.unique()[1]: 100000})
        X_train, y_train = smote.fit_resample(X_train.values.reshape(-1, 1), y_train)
        print(y_train.value_counts())
        X_train = pd.Series(X_train.flatten())
        X_test = pd.Series(X_test.values.flatten())

        X_train, y_train = build_frames(X_train, y_train)
        X_test, y_test = build_frames(X_test, y_test.values)

        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)
        pathlib.Path("data/preprocessed/road/").mkdir(parents=True, exist_ok=True)
        np.save(f"data/preprocessed/road/{attack_type}_features_train.npy", X_train)
        np.save(f"data/preprocessed/road/{attack_type}_labels_train.npy", y_train)
        np.save(f"data/preprocessed/road/{attack_type}_features_test.npy", X_test)
        np.save(f"data/preprocessed/road/{attack_type}_labels_test.npy", y_test)
    return X_train, X_test, y_train, y_test


def load_data_in_vehicle(car_brand, label):
    if os.path.exists(
        f"data/preprocessed/in_vehicle/{car_brand}_{label}_features.npy"
    ) and os.path.exists(f"data/preprocessed/in_vehicle/{car_brand}_{label}_labels.npy"):
        print("Loading Preprocessed Data")
        X = np.load(f"data/preprocessed/in_vehicle/{car_brand}_{label}_features.npy")
        y = np.load(f"data/preprocessed/in_vehicle/{car_brand}_{label}_labels.npy")
    else:
        data = pd.read_csv(f"data/in-vehicle/in-vehicle_{car_brand}.csv")

        y = data["Label"]
        y = y.apply(lambda x: 0.0 if x != label else x)
        X = data["ID"].apply(lambda x: int(x, 16))
        X, y = build_frames(X, y)

        X, y = shuffle(X, y, random_state=42)
        pathlib.Path("data/preprocessed/in_vehicle/").mkdir(parents=True, exist_ok=True)
        np.save(f"data/preprocessed/in_vehicle/{car_brand}_{label}_features.npy", X)
        np.save(f"data/preprocessed/in_vehicle/{car_brand}_{label}_labels.npy", y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test


def load_hcrl(attack_type):
    filenames = {
        "dos_data": "DoS_dataset.csv",
        "fuzzy_data": "Fuzzy_dataset.csv",
        "spoof_gear_data" : "gear_dataset.csv",
        "spoof_rpm_data": "RPM_dataset.csv",
    }
    if os.path.exists(
        f"data/preprocessed/car-hacking-dataset/{attack_type}_features.npy"
    ) and os.path.exists(
        f"data/preprocessed/car-hacking-dataset/{attack_type}_labels.npy"
    ):
        X = np.load(f"data/preprocessed/car-hacking-dataset/{attack_type}_features.npy")
        y = np.load(f"data/preprocessed/car-hacking-dataset/{attack_type}_labels.npy")
    else:
        data = pd.read_csv(
            f"data/car-hacking-dataset/{filenames[attack_type]}",
            names=[
                "timestamp",
                "ID",
                "DLC",
                "DATA0",
                "DATA1",
                "DATA2",
                "DATA3",
                "DATA4",
                "DATA5",
                "DATA6",
                "DATA7",
                "label",
            ],
        )
        data["ID"] = data["ID"].apply(lambda x: int(x, 16))
        X, y = build_frames(data["ID"], data["label"])
        X, y = shuffle(X, y, random_state=42)
        pathlib.Path("data/preprocessed/car-hacking-dataset/").mkdir(parents=True, exist_ok=True)
        np.save(f"data/preprocessed/car-hacking-dataset/{attack_type}_features.npy",X)
        np.save(f"data/preprocessed/car-hacking-dataset/{attack_type}_labels.npy",y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test
