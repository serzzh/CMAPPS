import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

MAXLIFE = 120
SCALE = 1
RESCALE = 1
true_rul = []
test_engine_id = 0
training_engine_id = 0


def kink_RUL(cycle_list, max_cycle):
    '''
    Piecewise linear function with zero gradient and unit gradient

            ^
            |
    MAXLIFE |-----------
            |            \
            |             \
            |              \
            |               \
            |                \
            |----------------------->
    '''
    knee_point = max_cycle - MAXLIFE
    kink_RUL = []
    stable_life = MAXLIFE
    for i in range(0, len(cycle_list)):
        if i < knee_point:
            kink_RUL.append(MAXLIFE)
        else:
            tmp = kink_RUL[i - 1] - (stable_life / (max_cycle - knee_point))
            kink_RUL.append(tmp)

    return kink_RUL


def compute_rul_of_one_id(FD00X_of_one_id, max_cycle_rul=None):
    '''
    Enter the data of an engine_id of train_FD001 and output the corresponding RUL (remaining life) of these data.
    type is list
    '''

    cycle_list = FD00X_of_one_id['cycle'].tolist()
    if max_cycle_rul is None:
        max_cycle = max(cycle_list)  # Failure cycle
    else:
        max_cycle = max(cycle_list) + max_cycle_rul
        # print(max(cycle_list), max_cycle_rul)

    # return kink_RUL(cycle_list,max_cycle)
    return kink_RUL(cycle_list, max_cycle)


def compute_rul_of_one_file(FD00X, id='engine_id', RUL_FD00X=None):
    '''
    Input train_FD001, output a list
    '''
    rul = []
    # In the loop train, each id value of the 'engine_id' column
    if RUL_FD00X is None:
        for _id in set(FD00X[id]):
            rul.extend(compute_rul_of_one_id(FD00X[FD00X[id] == _id]))
        return rul
    else:
        rul = []
        for _id in set(FD00X[id]):
            # print("#### id ####", int(RUL_FD00X.iloc[_id - 1]))
            true_rul.append(int(RUL_FD00X.iloc[_id - 1]))
            rul.extend(compute_rul_of_one_id(FD00X[FD00X[id] == _id], int(RUL_FD00X.iloc[_id - 1])))
        return rul


    
def data_norm(data):
    ### Standard Normal ###
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std.replace(0, 1, inplace=True)
    data = (data - mean) / std
    return data


def read_cmapss(url):
    df = pd.read_csv(url, header=None)
    columns_old = ["ENGINEID", "TIMECYCLE", "OPSET1", "OPSET2", "OPSET3", "Total temp at fan in (T2)", "Total temp at LPC out (T24)", "Total temp at HPC out (T30)", "Total temp at LPT out (T50)", 
"Pres at fan in (P2)", "Total pres in bypass-duct (P15)", "Total pres at HPC out (P30)", "Physical fan speed (Nf)", 
"Physical core speed (Nc)", "Engine pres ratio (epr=P50/P2)", "Static pres at HPC out (Ps30)", "Ratio of fuel flow to Ps30 (phi)",
"Corrected fan speed (NRf)", "Corrected core speed (NRc)", "Bypass Ratio (BPR)", "Burner fuel-air ratio (farB)", 
"Bleed Enthalpy (htBleed)", "Demanded fan speed (Nf_dmd)", "Demanded corrected fan speed (PCNfR_dmd)", "HPT coolant bleed (W31)",
"LPT coolant bleed (W32)", "FILEID","RUL"]
    columns_new = ["FILEID","ENGINEID", "TIMECYCLE", "OPSET1", "OPSET2", "OPSET3", "Total temp at fan in (T2)", "Total temp at LPC out (T24)", "Total temp at HPC out (T30)", "Total temp at LPT out (T50)", 
"Pres at fan in (P2)", "Total pres in bypass-duct (P15)", "Total pres at HPC out (P30)", "Physical fan speed (Nf)", 
"Physical core speed (Nc)", "Engine pres ratio (epr=P50/P2)", "Static pres at HPC out (Ps30)", "Ratio of fuel flow to Ps30 (phi)",
"Corrected fan speed (NRf)", "Corrected core speed (NRc)", "Bypass Ratio (BPR)", "Burner fuel-air ratio (farB)", 
"Bleed Enthalpy (htBleed)", "Demanded fan speed (Nf_dmd)", "Demanded corrected fan speed (PCNfR_dmd)", "HPT coolant bleed (W31)",
"LPT coolant bleed (W32)", "RUL"]
    df.columns = columns_old
    df = df[columns_new]

    return df

