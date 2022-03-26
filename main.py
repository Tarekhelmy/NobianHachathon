import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import json


Names=['valve (A) timestamp', 'valve (A) output value',
       'flow setpoint (B) timestamp', 'flow setpoint (B) value',
       'flow measurment (C) timestamp', 'flow measurment (C) value']


def filter(Signal,dt):
    order = 5
    sampling_freq = dt
    cutoff_freq = 100
    normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
    numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)
    filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, Signal)
    return filtered_signal

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
    new_data = new_data.drop(new_data[new_data['flow setpoint (B) value'] == 'Bad'].index)
    new_data = new_data.drop(new_data[new_data['flow measurment (C) value'] == 'Bad'].index)
    new_data = new_data.drop(data[data['valve (A) output value'] == 'Unknown'].index)
    new_data = new_data.drop(new_data[new_data['flow setpoint (B) value'] == 'Unknown'].index)
    new_data = new_data.drop(new_data[new_data['flow measurment (C) value'] == 'Unknown'].index)
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

def fastfourier(Signal,dt):
    Time_freq = np.fft.rfftfreq(len(Signal),dt)
    num_freq = np.fft.rfft(Signal)
    return Time_freq, num_freq

def get_classes(data,Start,End):
    A, TimeA = clean_data['valve (A) output value'][Start:End], clean_data['valve (A) timestamp'][Start:End]
    # B, TimeB = clean_data['flow setpoint (B) value'][Start:End], clean_data['flow setpoint (B) timestamp'][Start:End]
    C, TimeC = clean_data['flow measurment (C) value'][Start:End], clean_data['flow measurment (C) timestamp'][
                                                                   Start:End]
    TimeA = np.int64(TimeA)
    t0 = TimeA[0]
    TimeA = (TimeA - t0) / (10 ** 9)

    # TimeB = np.int64(TimeC)
    # TimeB = (TimeB - t0)/(10**9)

    TimeC = np.int64(TimeC)
    TimeC = (TimeC - t0) / (10 ** 9)

    t_span = np.linspace(TimeC[0], TimeC[-1], num=len(C))
    dt = t_span[1] - t_span[0]
    A = np.array(A, dtype=np.float64)
    # B = np.array(B, dtype=np.float64)
    C = np.array(C, dtype=np.float64)

    A = np.interp(t_span, TimeA, A)
    # B = np.interp(t_span, TimeB, B)
    C = np.interp(t_span, TimeC, C)

    # Finding start and end points of clogs
    clog_indicator = (A - C)
    clogs = np.where(clog_indicator > 50, 100, 0)
    clog_diff = np.diff(clogs)
    clog_start = np.where(clog_diff > 0, 150, 0)
    clog_end = np.where(clog_diff < 0, 150, 0)

    # Getting pre-clog data points
    delta = np.int64(350)
    clog_start_pts = np.nonzero(clog_start)[0]
    clog_end_pts = np.nonzero(clog_end)[0]
    clog_end_pts = np.insert(clog_end_pts, 0, 0)

    preclog_A = []
    # preclog_B = []
    preclog_C = []

    noclog_A = []
    # noclog_B = []
    noclog_C = []

    for idx, end_pt in enumerate(clog_start_pts):
        start_pt = end_pt - delta
        if (start_pt > 0) & (start_pt > clog_end_pts[idx]):
            preclog_A.append(filter(A[start_pt:end_pt], dt))
            # preclog_B.append(filter(B[start_pt:end_pt], dt))
            preclog_C.append(filter(C[start_pt:end_pt], dt))

        prev_start_pt = start_pt - delta
        if (start_pt > 0) & (prev_start_pt >= clog_end_pts[idx] + delta):
            noclog_A.append(filter(A[prev_start_pt:start_pt], dt))
            # noclog_B.append(filter(B[prev_start_pt:start_pt], dt))
            noclog_C.append(filter(C[prev_start_pt:start_pt], dt))

    preclog_A = np.array(preclog_A)
    # preclog_B = np.array(preclog_B)
    preclog_C = np.array(preclog_C)

    noclog_A = np.array(noclog_A)
    # noclog_B = np.array(noclog_B)
    noclog_C = np.array(noclog_C)

    class_1 = preclog_A - preclog_C
    class_2 = noclog_A - noclog_C

    return class_1, class_2

if __name__ == '__main__':
    data = pd.read_pickle('cached_dataframe.pkl')
    clean_data = drop_bad(data)
    end_indices = find_last_indices(clean_data)

    # Getting a Section of the data and interpolating to move all signals onto the same time axis
    Start = 0
    End = 710000
    class_1, class_2 = get_classes(clean_data, Start, End)
    pd.DataFrame(class_1).to_csv("clogged.csv")
    pd.DataFrame(class_2).to_csv("unclogged.csv")

    # Plotting Stuff
    # plt_end = -2
    #
    # plt.plot(t_span[0:plt_end], clog_start[0:plt_end+1], label='clog start')
    # plt.plot(t_span[0:plt_end], clog_end[0:plt_end+1], label='clog end')
    # # plt.plot(t_span[0:plt_end], A[0:plt_end], label='A')
    # # plt.plot(t_span[0:plt_end], C[0:plt_end], label='C')
    # # # plt.plot(t_span[0:-1], B[0:-1], label='B')
    # # plt.plot(t_span[0:-1], indicator_1[0:-1], label='clog_indicator')
    # plt.legend()
    # plt.show()