# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

End=[805440,846848]

Names=['valve (A) timestamp', 'valve (A) output value',
       'flow setpoint (B) timestamp', 'flow setpoint (B) value',
       'flow measurment (C) timestamp', 'flow measurment (C) value']

def ExcelPickler():
    csvreader = pd.read_excel('Hackathon_Nobian_dataset.xlsx')
    csvreader.to_pickle('cached_dataframe.pkl')
    return None

def Removebad():
    data = pd.DataFrame(pd.read_pickle('cached_dataframe.pkl'))
    indexs= []
    A, TimeA = data['valve (A) output value'], data['valve (A) timestamp']
    C, TimeC = data['flow measurment (C) value'], data['flow measurment (C) timestamp']
    B, TimeB = data['flow setpoint (B) value'], data['flow setpoint (B) timestamp']
    for i in range(len(A)):
        if A[i]=="Bad":
            indexs.append(i)
    for i in range(len(C)):
        if C[i]=="Bad":
            indexs.append(i)
    for i in range(len(B)):
        if B[i]=="Bad":
            indexs.append(i)
    data=data.drop(indexs)
    data.to_pickle('cached_dataframe_edited.pkl')


def Plot():
    data=pd.read_pickle('cached_dataframe_edited.pkl')
    plt.plot(data['valve (A) timestamp'][0:10000],data['valve (A) output value'][0:10000],label="A")
    plt.plot(data['flow measurment (C) timestamp'][0:10000],data['flow measurment (C) value'][0:10000],label="C")
    plt.legend()
    plt.show()
    print(data.keys())

def ROC(Start,End):
    data = pd.read_pickle('cached_dataframe_edited.pkl')
    A ,TimeA =  data['valve (A) output value'][Start:End[0]] ,data['valve (A) timestamp'][Start:End[0]]
    C ,TimeC =  data['flow measurment (C) timestamp'][Start:End[-1]] ,data['flow measurment (C) value'][Start:End[-1]]
    A = A.to_numpy()
    A = np.asarray(A)
    C = C.to_numpy()
    C = np.asarray(A)
    Errors = 0
    Adiff = np.diff(A)
    Cdiff = np.diff(C)
    Aedit = A[::5]
    Cedit = C[::5]
    for i in range(len(Aedit)):
        if A[i]<5:
            Errors+=1
    print(Errors)
    return None


if __name__ == '__main__':
    # ROC(0,End)
    # Removebad()
    Plot()
