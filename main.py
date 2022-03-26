import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


Names=['valve (A) timestamp', 'valve (A) output value',
       'flow setpoint (B) timestamp', 'flow setpoint (B) value',
       'flow measurment (C) timestamp', 'flow measurment (C) value']

def ExcelPickler():
    csvreader = pd.read_excel('Hackathon_Nobian_dataset.xlsx')
    csvreader.to_pickle('cached_dataframe.pkl')
    return None

def Plot(data):
    plt.plot(data['valve (A) timestamp'][0:10000],data['valve (A) output value'][0:10000],label="A")
    plt.plot(data['flow measurment (C) timestamp'][0:10000],data['flow measurment (C) value'][0:10000],label="C")
    plt.legend()
    plt.show()
    print(data.keys())

# Result of Last Indices[805440, 644258, 846848]

def find_last_indices(data):
    last_index = []
    last_index.append(data['valve (A) timestamp'].last_valid_index())
    last_index.append(data['flow setpoint (B) timestamp'].last_valid_index())
    last_index.append(data['flow measurment (C) timestamp'].last_valid_index())
    return last_index

def drop_bad(data):
    new_data = data.drop(data[data['valve (A) output value'] == 'Bad'].index)
    new_data = new_data.drop(data[data['flow setpoint (B) value'] == 'Bad'].index)
    new_data = new_data.drop(data[data['flow measurment (C) value'] == 'Bad'].index)
    return new_data

def ROC(data,Start,End):
    A, TimeA =  data['valve (A) output value'][Start:End[0]],data['valve (A) timestamp'][Start:End[0]]
    C, TimeC =  data['flow measurment (C) timestamp'][Start:End[2]] ,data['flow measurment (C) value'][Start:End[2]]
    A = A.to_numpy()
    A = np.asarray(A)
    Errors = 0
    Adiff = np.diff(A)
    Cdiff = np.diff(C)
    Aedit = A[::5]
    for i in range(len(Aedit)):
        if A[i]<5:
            Errors+=1
    print(Errors)
    return None

# def clog_finder(data,start,end):
#
#     return
if __name__ == '__main__':
    data = pd.read_pickle('cached_dataframe.pkl')
    clean_data = drop_bad(data)
    end_indices = find_last_indices(clean_data)
    # ROC(clean_data, 0, end_indices)
    # Plot(clean_data)
    Start = 0
    End = 10000
    A, TimeA = clean_data['valve (A) output value'][Start:End], clean_data['valve (A) timestamp'][Start:End]
    C, TimeC = clean_data['flow measurment (C) value'][Start:End], clean_data['flow measurment (C) timestamp'][Start:End]
    TimeA = np.int64(TimeA)
    t0 = TimeA[0]
    TimeA = (TimeA - t0)/(10**9)

    TimeC = np.int64(TimeC)
    TimeC = (TimeC - t0)/(10**9)

    t_span = np.linspace(TimeA[0], TimeA[-1], num=len(TimeC))
    A = np.array(A, dtype=np.float64)
    C = np.array(C, dtype=np.float64)
    A = np.interp(t_span, TimeA, A)
    C = np.interp(t_span, TimeC, C)

    difference = A-C
    plt.plot(t_span[0:-1], np.diff(difference))
    plt.plot(t_span[0:-1], A[0:-1], label = 'A')
    plt.plot(t_span[0:-1], C[0:-1], label = 'C')
    plt.show()
