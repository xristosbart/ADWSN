# Filename: load_lab_data.py
# Author: Christos Gklezos

import torch
import pandas as pd
import numpy as np
import os

DEVICES = {"f6ce36f0118e6361":0, "f6ce36d563cef9cb":1, "f6ce364ff4c1c55a":2, "f6ce368d7563b285":3, "f6ce3667a3445b20":4, "f6ce36c1896a819b":5, "f6ce36ef672a639d":6, "f6ce368f8d612db5":7}


def load_file(file):
    data = pd.read_csv(file)
    return data


def load_dir(folder):
    files = os.listdir(folder)
    data = pd.DataFrame()
    for file in files:
        frame = load_file(folder+file)
        data = data.append(frame)
    return data


def genTrainData(folder, feature):
    d = load_dir(folder)
    d_feature_only = d[d['Sensor'] == feature]
    DT = np.empty(8)
    FT = np.empty(8)
    out = pd.DataFrame(columns=range(8))

    for k, v in DEVICES.items():
        DT[v] = 0
        FT[v] = 0

    for row in d_feature_only.iterrows():
        dev = row[1]["DeviceId"]
        if FT[DEVICES[dev]] == 0:
            FT[DEVICES[dev]] = 1
            DT[DEVICES[dev]] = row[1]["Value"]
        else:  # == 1
            for key, val in DEVICES.items():
                if FT[val] == 0:
                    for k, v in DEVICES.items():
                        DT[v] = 0
                        FT[v] = 0
                    FT[DEVICES[dev]] = 1
                    DT[DEVICES[dev]] = row[1]["Value"]
                    break
            else:
                # All values are filled
                out = out.append(pd.Series(DT), ignore_index=True)
                for k, v in DEVICES.items():
                    DT[v] = 0
                    FT[v] = 0
                FT[DEVICES[dev]] = 1
                DT[DEVICES[dev]] = row[1]["Value"]
    return torch.tensor(out.to_numpy())


t_data = genTrainData("LabData/20220314_Week/", 'Temperature')

torch.save(t_data, "lab_sample_data_2.pt")


