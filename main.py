# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
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
    data=pd.read_pickle('cached_dataframe.pkl')
    plt.plot(data['valve (A) timestamp'][20:10000],data['valve (A) output value'][20:10000])
    plt.show()
    print(data.keys())

def ROC(Start,End):
    data = pd.read_pickle('cached_dataframe_edited.pkl')
    A ,TimeA =  data['valve (A) output value'][Start:End] ,data['valve (A) timestamp'][Start:End]
    C ,TimeC =  data['flow measurment (C) timestamp'][Start:End] ,data['flow measurment (C) value'][Start:End]
    A = A.to_numpy()
    A = np.asarray(A)
    for i in range(len(A)):
        print(A[i])
        A[i]=float(A[i])
    Errors = 0
    Adiff = np.diff(A)
    Cdiff = np.diff(C)
    Aedit = A[::5]
    for i in range(len(Aedit)):
        if A[i]<5:
            Errors+=1
    print(Errors)
    return None


if __name__ == '__main__':
    ROC(0,600000)
    # Removebad()
    # Plot()
