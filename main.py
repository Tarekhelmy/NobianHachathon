# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import matplotlib.pyplot as plt
import json


Names=['valve (A) timestamp', 'valve (A) output value',
       'flow setpoint (B) timestamp', 'flow setpoint (B) value',
       'flow measurment (C) timestamp', 'flow measurment (C) value']

def ExcelPickler():
    csvreader = pd.read_excel('Hackathon_Nobian_dataset.xlsx')
    csvreader.to_pickle('cached_dataframe.pkl')
    return None

def Plot():
    data=pd.read_pickle('cached_dataframe.pkl')
    for key in data.keys():
        for i in range(len(data[key])):
            if not data['valve (A) output value'][i].isalpha():
                print(i)
    plt.plot(data['valve (A) timestamp'][20:10000],data['valve (A) output value'][20:10000])
    plt.show()
    print(data.keys())


if __name__ == '__main__':
    Plot()
