import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions( linewidth=100)
def parse_data():
    df = pd.read_csv('eth.csv')
    extra_columns = ['date', 'BlkCnt', 'BlkSizeByte', 'BlkSizeMeanByte', 'FeeMeanNtv', 'FeeMedNtv', 'FeeTotNtv', 'DiffMean', \
                      'FeeMeanUSD', 'FeeMedUSD', 'IssContPctAnn', 'IssTotUSD', 'NVTAdj', 'NVTAdj90', 'IssContNtv', 'IssContUSD', \
                      'PriceBTC', 'TxCnt', 'TxTfr', 'TxTfrValMedNtv', 'TxTfrValMedUSD', 'TxTfrValNtv', \
                      'TxTfrValUSD', 'VtyDayRet180d', 'VtyDayRet30d', 'VtyDayRet60d', 'ROI1yr', 'ROI30d']
    df.drop(columns=extra_columns, inplace=True)
    df = df.iloc[1000:]
    df = df.to_numpy()
    return(df)

def scale_data(data):
    max = np.amax(data, axis = 0)
    return (data / max), max

df = parse_data()
df, max = scale_data(df)
print(df)