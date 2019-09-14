import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
np.set_printoptions( linewidth=100)
def parse_data():
    df = pd.read_csv('eth.csv')
    extra_columns = ['date', 'BlkCnt', 'BlkSizeByte', 'BlkSizeMeanByte', 'FeeMeanNtv', 'FeeMedNtv', 'FeeTotNtv', 'DiffMean', \
                      'FeeMeanUSD', 'FeeMedUSD', 'IssContPctAnn', 'IssTotUSD', 'NVTAdj', 'NVTAdj90', 'IssContNtv', 'IssContUSD', \
                      'PriceBTC', 'TxCnt', 'TxTfr', 'TxTfrValMedNtv', 'TxTfrValMedUSD', 'TxTfrValNtv', \
                      'TxTfrValUSD', 'VtyDayRet180d', 'VtyDayRet30d', 'VtyDayRet60d', 'ROI1yr', 'ROI30d']
    df.drop(columns=extra_columns, inplace=True)
    #even_more = ['AdrActCnt', 'SplyCur', 'FeeTotUSD', 'IssTotNtv', 'TxTfrValAdjNtv', 'TxTfrValMeanNtv', 'TxTfrValMeanUSD']
    #df.drop(columns=even_more, inplace=True)
    df = df.iloc[-768:]
    col = df.columns
    df = df.to_numpy()
    return(df, col)

def scale_data(data):
    max = np.amax(data, axis = 0)
    return (data / max), max

def split_data(data, fraction_test):
    fraction_training = 1  -fraction_test
    split_index = int(num_points(data) * fraction_training)
    train, test = np.split(data, [split_index])
    trainy = train[:,[4]]
    testy = test[:,[4]]
    return (train, test, trainy, testy)


def num_points(data):
    return data.shape[0]

def num_inputs(data):
    return data.shape[1]

def plot_data(data, columns, col=[]):
    if col == []:
        #print all columns
        fig, axs = plt.subplots(len(columns))
        for i in range(len(columns)):
            axs[i].plot(data[:,i])
            axs[i].set_title(columns[i])
    else:
        #print specific columns
        fig, axs = plt.subplots(len(col))
        j = 0
        for i in range(len(columns)):
            if columns[i] in col:
                axs[j].plot(data[:,i])
                axs[j].set_title(columns[i], y=0.5, loc='right')
                j += 1
    plt.show()
    
