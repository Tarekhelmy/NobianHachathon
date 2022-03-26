# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import datetime
import json

End=[200000,200000]

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

def drop_bad(data):
    new_data = data.drop(data[data['valve (A) output value'] == 'Bad'].index)
    new_data = new_data.drop(data[data['flow setpoint (B) value'] == 'Bad'].index)
    new_data = new_data.drop(data[data['flow measurment (C) value'] == 'Bad'].index)
    return new_data

def Plot():
    data=pd.read_pickle('cached_dataframe.pkl')
    plt.plot(data['valve (A) timestamp'][0:10000],data['valve (A) output value'][0:10000],label="A")
    plt.plot(data['flow measurment (C) timestamp'][0:10000],data['flow measurment (C) value'][0:10000],label="C")
    plt.legend()
    plt.show()
    print(data.keys())

def ROC(Start,End):
    data = pd.read_pickle('cached_dataframe_edited.pkl')
    data = drop_bad(data)
    A ,TimeA =  data['valve (A) output value'][Start:End[0]] ,data['valve (A) timestamp'][Start:End[0]]
    C ,TimeC =  data['flow measurment (C) value'][Start:End[-1]] ,data['flow measurment (C) timestamp'][Start:End[-1]]
    TimeA = np.int64(TimeA)
    t0 = TimeA[0]
    TimeA = (TimeA - t0)/(10**9)
    TimeC = np.int64(TimeC)
    TimeC = (TimeC - t0)/(10**9)
    A = np.array(A, dtype=np.float64)
    C = np.array(C, dtype=np.float64)
    timeAinterp = np.linspace(TimeC[0] , TimeC[-1],len(C))
    dt = (timeAinterp[1]-timeAinterp[0])
    Ainterp = np.interp(timeAinterp,TimeA,A)
    Cinterp = np.interp(timeAinterp,TimeC,C)
    diff = Ainterp - Cinterp
    clogpoint = np.where(diff>50,200,0)
    clogstart = np.where(np.diff(clogpoint)>0,200,0)
    index = np.arange(len(clogstart))
    index = index[clogstart>0]
    var = 350
    arraystime,arraysvalueC,arraysvalueA,arraysdiff=  np.zeros((4,len(index),var))
    for i in range(len(arraystime)):
        arraysdiff[i] = diff[index[i]-var:index[i]]
        arraystime[i] = timeAinterp[index[i]-var:index[i]]
        arraysvalueC[i] = Cinterp[index[i]-var:index[i]]
        arraysvalueA[i] = Ainterp[index[i]-var:index[i]]
    for i in range(len(arraystime)):
        plt.plot(arraystime[i],filter(arraysdiff[i],arraystime[i],dt))
        plt.plot(arraystime[i],arraysdiff[i])
        plt.show()
    plt.plot(timeAinterp[:-1],clogstart,label="clogdetect")
    plt.plot(timeAinterp,diff,label="diff")
    plt.plot(timeAinterp,Ainterp,label="A")
    plt.plot(timeAinterp,Cinterp,label="C")
    plt.legend()
    # plt.show()
    return None

def filter(Signal,Time,dt):
    order = 5
    sampling_freq = dt
    cutoff_freq = 100
    sampling_duration = Time[-1]
    time = Time
    normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
    numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)
    filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, Signal)
    return filtered_signal

if __name__ == '__main__':
    ROC(0,End)
    # ExcelPickler()
    # Removebad()
    # Plot()
