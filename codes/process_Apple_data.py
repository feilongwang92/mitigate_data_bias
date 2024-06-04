
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys, os

### ------------------ use applemobilitytrends.csv ---------------
# data is acquired by finding out the king-county row from applemobilitytrends.csv, copy it and reverse it into column
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends.csv')

df = rawdata.loc[rawdata['region'] == "King County"]
df = df.loc[df['transportation_type'] == "transit"]

df = df.iloc[:,6:].T #.to_frame() # take the time series only (which starts from the 6th columns)
df.columns = ['transit']
df.dtypes

# rawdata = pd.read_csv(r'D:\W_ProgamData\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_king_transit_daily.csv')

df['date'] = pd.to_datetime(df.index) #.dt.date
df.set_index('date', inplace=True)

df[['transit']] = df[['transit']] - 93.6 # 100 # 93.6 is the mean of Jan 2020
# rawdata = rawdata[['transit']]
df_month = df.resample("M").mean()


## New York City
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends.csv')

df = rawdata.loc[rawdata['region'].isin(['Bronx County', 'Kings County', 'New York County', 'Queens County', 'Richmond County'])]
df = df.loc[df['transportation_type'] == "transit"]

df = df.iloc[:,6:].T #.to_frame() # take the time series only (which starts from the 6th columns)
df.columns = ['Bronx County', 'Kings County', 'New York County', 'Queens County', 'Richmond County']
df.loc[['2021-03-12'],:] = (df.loc[['2021-03-05']].values+df.loc[['2021-03-19']].values)/2 # NaN at '2022-03-21'
df.loc[['2022-03-21'],:] = (df.loc[['2022-03-14']].values+df.loc[['2022-03-28']].values)/2 # NaN at '2022-03-21'

# population of the 5 counties: np.array([1.427, 2.577, 1.629, 2.271, 0.4756]).sum() = 8.3796
df['transit_mean5county'] = (df[['Bronx County']].values*1.427 + df[['Kings County']].values*2.577+ df[['New York County']].values*1.629
 + df[['Queens County']].values*2.271+ df[['Richmond County']].values*0.4756)/8.3796

df = df[['transit_mean5county']]

df.dtypes

# rawdata = pd.read_csv(r'D:\W_ProgamData\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_king_transit_daily.csv')

df['date'] = pd.to_datetime(df.index) #.dt.date
df.set_index('date', inplace=True)

df[['transit_mean5county']] = df[['transit_mean5county']] - 98.45 # 100 # 93.6 is the mean of Jan 2020
# rawdata = rawdata[['transit']]
df_month = df.resample("M").mean()

## Washington Metropolitan Area
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends.csv')
countiesInWMA = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\counties_WMA_wiki.csv')
countiesInWMA.iloc[3,:] = ['Washington DC','Washington DC',705749.0]
countiesInWMA.iloc[9,1] = 'Alexandria City'

[0,1,2,3,7,9] # other rows (regions) do not have "transit" data
regions = ['Fairfax County', 'Montgomery County', 'Prince George\'s County','Washington DC', 'Arlington County', 'Alexandria City']
df = rawdata.loc[rawdata['region'].isin(regions)]
df = df.loc[df['transportation_type'] == "transit"]

df = df[~df['sub-region'].isin(['Ohio', 'Pennsylvania'])] # the two State also have 'Montgomery County'
countyNames = df['region'].values

df = df.iloc[:,6:].T #.to_frame() # take the time series only (which starts from the 6th columns)
df.columns = countyNames
df.loc[['2021-03-12'],:] = (df.loc[['2021-03-05']].values+df.loc[['2021-03-19']].values)/2 # NaN at '2022-03-21'
df.loc[['2022-03-21'],:] = (df.loc[['2022-03-14']].values+df.loc[['2022-03-28']].values)/2 # NaN at '2022-03-21'

df.loc[['2020-05-11'],'Washington DC'] = (df.loc[['2020-05-04'],'Washington DC'].values+df.loc[['2020-05-18'],'Washington DC'].values)/2
df.loc[['2020-05-12'],'Washington DC'] = (df.loc[['2020-05-05'],'Washington DC'].values+df.loc[['2020-05-19'],'Washington DC'].values)/2

# population of the 5 counties: np.array([1.427, 2.577, 1.629, 2.271, 0.4756]).sum() = 8.3796
countyPopulation = dict(zip(countiesInWMA['County'], countiesInWMA['population (2019 Estimate)']))

# for i in regions: df[i] = df[i].values*countyPopulation[i]
# df['transit_apple'] = df[regions].sum(axis=1)/sum([countyPopulation[i] for i in regions])

df['transit_apple'] = np.sum([df[i].values*countyPopulation[i] for i in regions], axis=0)/sum([countyPopulation[i] for i in regions])

df = df[['transit_apple']]

df.dtypes

df['date'] = pd.to_datetime(df.index) #.dt.date
df.set_index('date', inplace=True)

df[['transit_apple']] = df[['transit_apple']] - 97.349 # 100 # 93.6 is the mean of Jan 2020

df_month = df.resample("M").mean()

df.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_WMA_transit_daily_change.csv') # , index=False
df_month.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_WMA_transit_monthly_change.csv')

##### Los Angeles County
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends.csv')
df = rawdata.loc[rawdata['region']=='Los Angeles County']
df = df.loc[df['transportation_type'] == "transit"]

df = df.iloc[:, 6:].T #.to_frame() # take the time series only (which starts from the 6th columns)
df.columns = ['transit_apple']
df.loc[['2021-03-12'],:] = (df.loc[['2021-03-05']].values+df.loc[['2021-03-19']].values)/2 # NaN at '2022-03-21'
df.loc[['2022-03-21'],:] = (df.loc[['2022-03-14']].values+df.loc[['2022-03-28']].values)/2 # NaN at '2022-03-21'

df['date'] = pd.to_datetime(df.index) #.dt.date
df.set_index('date', inplace=True)

# one type of baseline
df.resample("M").mean() # the mean of the first month: 103.140000

df[['transit_apple']] = df[['transit_apple']] - 103.14 # 103.14 is the mean of Jan 2020
df.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_LAC_transit_daily_change.csv') # , index=False
df_month.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_LAC_transit_monthly_change.csv')

## Try another type of baseline
baseline4week = df['transit_apple'].values[:28]
baseline4week = baseline4week.reshape([4,-1])
baseline4week = np.median(baseline4week,axis=0)

for i in range(28, df.shape[0]):
    df.iloc[i,0] -= baseline4week[(i-28)%4] # - baseline
df = df.iloc[28:,]

df_month = df.resample("M").mean()

df.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_LAC_transit_daily_change1.csv') # , index=False
df_month.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_LAC_transit_monthly_change1.csv')





"""
   ##  transit data
"""
# New york MTA
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\MTA_Daily_Ridership_Data__Beginning_2020.csv')

rawdata['Date'] = pd.to_datetime(rawdata["Date"])

rawdata.set_index('Date', inplace=True)
rawdata.columns
# 'Date', 'Subways: Total Estimated Ridership',
#        'Subways: % of Comparable Pre-Pandemic Day',
#        'Buses: Total Estimated Ridership',
#        'Buses: % of Comparable Pre-Pandemic Day'
rawdata['transit'] = rawdata['Subways: Total Estimated Ridership'].values + rawdata['Buses: Total Estimated Ridership'].values

df = (rawdata[['transit']] - 3233694.8) / 3233694.8 * 100 # 3233694.8 mean of first 7 days

df_month = df.resample("M").mean()

# Washington Metropolitan Area
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\WashingtonMetropolitanAreaTransitAuthority.csv')
rawdata['date'] = pd.to_datetime(rawdata["date"])
rawdata.set_index('date', inplace=True)
df = rawdata[['transit_agency']]

df.resample("M").mean() # mean of the first month: 771169.892939

df = (df[['transit_agency']] - 771169.892939) / 771169.892939 * 100

df = df.resample("M").mean()

df.to_csv(r'D:\W_ProgamData\Python\covid_w\compareBigAndSmall\agencyData\WashingtonMetropolitanAreaTransitAuthority_monthly_change.csv')
