
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys, os

sys.path.extend(['C:\\Users\\wangf\\Python\\covid_w\\compareBigAndSmall\\bayesian_changepoint_detection-master'])
from bayesian_changepoint_detection.generate_data import generate_normal_time_series
from bayesian_changepoint_detection.priors import const_prior
from functools import partial
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
import bayesian_changepoint_detection.offline_likelihoods as offline_ll


"""
   ## Transit App data: 
   https://docs.google.com/spreadsheets/d/1mSP9EXIhWqPQHOirfspSk1Q9ta01fkHvBuppQl0invQ/edit#gid=401959734
"""
## Seattle
# agency
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\King_KCM_ST_transit_monthly.csv')
data = agencymonthly.iloc[3:,]['KCM_ST'].values.reshape(-1,1)
agency_King = agencymonthly.iloc[3:3+25,1].to_list()

# transitAPP
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\transitApp-king-daily.csv')

rawdata.dtypes
rawdata['Date'] = pd.to_datetime(rawdata["Date"]) #.dt.date

rawdata.set_index('Date', inplace=True)
rawdata.columns

# rawdata[['transit']] = rawdata[['transit']] - 93.6 # 100 # 93.6 is the mean of Jan 2020
rawdata = rawdata[['TransitApp-KCM']]
rawdata_month = rawdata.resample("M").mean()

# compute correlation
transitApp_King = rawdata_month.iloc[2:2+25,0].to_list()
print(np.corrcoef(transitApp_King, agency_King))

# change point detection
data = rawdata_month.iloc[2:2+25,]['TransitApp-KCM'].values.reshape(-1,1)
data.shape

mean_data = data.mean(axis=0)
std_data = data.std(axis=0)
data = data - mean_data # df_data.subtract(mean_data, axis=1)
data = data / std_data # df_data.div(std_data, axis=1)
for i in data: print(i[0])

prior_function = partial(const_prior, p=0.01)#p=0.1 # p=1/(len(data) + 1)

Q, P, Pcp = offline_changepoint_detection(data, prior_function ,offline_ll.StudentT(), truncate=-40)

len(Pcp)
for p_i in np.exp(Pcp).sum(0): print(p_i)


#%%
"""
    New York City
"""
## NYC
# agency
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\NYC_transit_monthly.csv')
agency_NYC = agencymonthly.iloc[1:1+25,1].to_list()

for i in agencymonthly['transit'].to_list(): print(i)

# transitAPP
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\transitApp-NYC-daily.csv')
rawdata.dtypes
# rawdata['TransitApp-NYC_p'] = rawdata['TransitApp-NYC']
# rawdata['TransitApp-NYC'] = rawdata['TransitApp-NYC_p'].str.rstrip('%').astype('int')
rawdata['Date'] = pd.to_datetime(rawdata["Date"]) #.dt.date
rawdata.set_index('Date', inplace=True)
rawdata.columns
rawdata = rawdata[['TransitApp-NYC']]
rawdata_month = rawdata.resample("M").mean()

# compute corr
transitApp_NYC = rawdata_month.iloc[2:2+25,0].to_list()
print(np.corrcoef(transitApp_NYC, agency_NYC))

for i in rawdata_month['TransitApp-NYC'].to_list():
    print(i)

# change point detection
# data = rawdata_month.iloc[2:2+25,]['TransitApp-NYC'].values.reshape(-1,1)
data = rawdata_month.iloc[3:3+25,]['TransitApp-NYC'].values.reshape(-1,1) # for change point detection

mean_data = data.mean(axis=0)
std_data = data.std(axis=0)
data = data - mean_data # df_data.subtract(mean_data, axis=1)
data = data / std_data # df_data.div(std_data, axis=1)
for i in data: print(i[0])

prior_function = partial(const_prior, p=0.01)#p=0.1 # p=1/(len(data) + 1)

Q, P, Pcp = offline_changepoint_detection(data, prior_function ,offline_ll.StudentT(), truncate=-40)

len(Pcp)
for p_i in np.exp(Pcp).sum(0): print(p_i)

"""
   scatter plot
"""
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\transitApp-king-monthly.csv')
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\King_KCM_ST_transit_monthly.csv')

rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\transitApp-NYC-monthly.csv')
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\NYC_transit_monthly.csv')

rawdata.dtypes
rawdata['Date'] = pd.to_datetime(rawdata["Date"]).dt.strftime('%m/%Y')
rawdata.columns = ['date', 'transit_google']
agencymonthly['date'] = pd.to_datetime(agencymonthly["date"]).dt.strftime('%m/%Y')
agencymonthly.columns = ['date', 'transit_agency']
# googlemonthly_seattle.set_index('date', inplace=True)
# googlemonthly_seattle.columns

# general comparison
googleAppleAgency = pd.merge(rawdata, agencymonthly, on='date')
googleAppleAgency = googleAppleAgency.iloc[2:]
google_sub = googleAppleAgency['transit_google'].values
agency_sub = googleAppleAgency['transit_agency'].values

np.mean(google_sub), np.std(google_sub)
np.mean(agency_sub), np.std(agency_sub)

cumm_deviation_google_mean = [np.mean(google_sub[:i]-agency_sub[:i]) for i in range(1, len(google_sub)+1)]
for i in cumm_deviation_google_mean: print(i)

## agency vs google
agencyVSgoogle_King = pd.merge(agencymonthly, rawdata,on='date')
agencyVSgoogle_King['monthSince'] = agencyVSgoogle_King.index
agencyVSgoogle_King.set_index('date', inplace=True)

agencyVSgoogle_King = agencyVSgoogle_King.iloc[2:]
m, b = np.polyfit(agencyVSgoogle_King['transit_agency'], agencyVSgoogle_King['transit_google'], 1) #0.95, 17
alphas = np.linspace(0.1, 1, agencyVSgoogle_King.shape[0])
plt.figure(figsize=(4,3))
plt.scatter(agencyVSgoogle_King['transit_agency'], agencyVSgoogle_King['transit_google'], marker='s', alpha=alphas, color='darkred')
plt.xlabel('Agency'), plt.ylabel('TransitApp') #, plt.title('King County')
# plt.xlim((-90,40)), plt.ylim((-90,40))
plt.plot(agencyVSgoogle_King['transit_agency'], m*agencyVSgoogle_King['transit_agency']+b, color='red')
plt.tight_layout()
plt.savefig("TransitAppScatter_KC.svg")
plt.show()