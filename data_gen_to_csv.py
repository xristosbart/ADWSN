# data_gen_to_csv.py
# Author: Christos Gklezos


import torch
import pandas as pd
import math


def createDayTemperatureSeries(mean, amp, phase):
    return [mean - amp * math.cos((2 * math.pi * t / 24) + phase) for t in range(0, 24)]

def toCSV(series, filename):
    series = torch.tensor(series)
    series = series.numpy()
    series = pd.DataFrame(series)
    series.to_csv(filename, header=False)

def to_C_array(series_list, filename):
    f = open(filename,"w")
    f.write("float SERIES_TABLE["+str(len(series_list))+"]["+str(len(series_list[0]))+"] = {")
    for series in series_list:
        f.write("\n{")
        for val in series:
            f.write(str(val)+", ")
        f.truncate(f.tell()-2)
        f.seek(0, 2)
        f.write("},")
    f.truncate(f.tell() - 1)
    f.seek(0, 2)
    f.write("\n};")


def main():
    mean = 15
    amp = 5
    phase = 0 # ex. alt.: -3*2*math.pi/24
    filename = "temp_series.csv"

    tempseries = createDayTemperatureSeries(mean, amp, phase)
    #toCSV(tempseries, filename)
    to_C_array([tempseries, tempseries], "series.c")
    return 0


if __name__ == "__main__":
    main()
