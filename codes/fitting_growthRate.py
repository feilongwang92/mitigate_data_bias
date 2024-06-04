import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys, os
import scipy.optimize as opt
import pickle

# os.chdir('C:\Users\wangf\Python\Covid-19\googleData')

### ------------------ use applemobilitytrends.csv ---------------
# ### data is acquired by finding out the king-county row from applemobilitytrends.csv, copy it and reverse it into column
# rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends.csv')
#
# rawdata = rawdata.loc[(rawdata['country'] == "United States") & (rawdata['geo_type'] == "county")]
# rawdata = rawdata.loc[rawdata['transportation_type'] == "transit"]
# rawdata = rawdata.drop(['geo_type', 'transportation_type', 'alternative_name', 'country'], axis=1)
#
# rawdata.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\appleTransitTrends_UScounty.csv',index=False)

### Start here
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\appleTransitTrends_UScounty.csv')
rawdata.shape  # (152, 823) # far smaller than the 3000 counties in the country

## check missing values
rawdata.columns[rawdata.isna().any()].tolist()  # ['2021-03-12', '2022-03-21']
rawdata.loc[:, ['2021-03-12']] = (rawdata.loc[:, ['2021-03-05']].values +
                                  rawdata.loc[:, ['2021-03-19']].values) / 2  # NaN at '2022-03-21'
rawdata.loc[:, ['2022-03-21']] = (rawdata.loc[:, ['2022-03-14']].values +
                                  rawdata.loc[:, ['2022-03-28']].values) / 2  # NaN at '2022-03-21'

df = rawdata.iloc[:, 6:]  # .to_frame() # take the time series only (which starts from the 6th columns)
mean_Jan = np.array(df.iloc[:, :15].mean(axis=1))  # calculate mean value in January
len(mean_Jan)

# recalculate using the new baseline
df = df.subtract(mean_Jan, axis=0)
df = df.T

# set date_str to datatime
df['date'] = pd.to_datetime(df.index)  # .dt.date
df.set_index('date', inplace=True)

# average to mean
df_month = df.resample("M").mean()

## plot
df_month.plot(alpha=0.3, linewidth=0.5, legend=False)
plt.ylim((-100, 100))
plt.ylabel("Change in transit usage (%)")

## standardization
mean_county = np.array(df_month.mean(axis=0))
std_county = np.array(df_month.std(axis=0))

df_month = df_month.subtract(mean_county, axis=1)
df_month = df_month.div(std_county, axis=1)

## plot
df_month.plot(alpha=0.3, linewidth=0.5, legend=False)
# plt.ylim((-100, 100))
plt.ylabel("Change in transit usage (Standardized)")


## Fitting a Logistic Curve to Data
# https://stackoverflow.com/questions/56329180/fitting-a-logistic-curve-to-data

def f(x, L, k, t0, b):
    return L / (1. + np.exp(-k * (x - t0))) + b


rawdata['region'] = rawdata[['region', 'sub-region']].agg(', '.join, axis=1)
county_list = rawdata['region'].to_list()  # any duplicates? set([x for x in county_list if county_list.count(x) > 1])
x_2fit = np.array(list(range(12))) + 1  # when fitting, better no 0 to avoid potential error

fitted_para = dict()
for i in range(df_month.shape[1]):
    y_std2fit = np.array(df_month.iloc[8:8 + 12, i])
    (L_, k_, t0_, b_), pcov = opt.curve_fit(f, x_2fit, y_std2fit, p0=[5, 1, 5, -1], maxfev=5000, method="trf")
    y_fitted = f(x_2fit, L_, k_, t0_, b_)
    residuals = y_std2fit - y_fitted
    ss_total = np.sum((y_std2fit - np.mean(y_std2fit)) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    # print(r_squared)

    # if 0 < k_ < 2.5:
    if r_squared > 0.7:
        fitted_para[county_list[i]] = [L_, k_, t0_, b_]
len(fitted_para)  # 127
for county_i in fitted_para.keys(): print(county_i, '\t', fitted_para[county_i][0],
                                          '\t', fitted_para[county_i][1], '\t', fitted_para[county_i][2],
                                          '\t', fitted_para[county_i][3])

with open(r"C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\Apple_fitted_para_R0.7.pkl", 'wb') as fp:
    pickle.dump(fitted_para, fp)

with open(r"C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\Apple_fitted_para_R0.7.pkl", 'rb') as f:
    fitted_para = pickle.load(f)

## histogram of fitted k
data = [fitted_para[c][1] for c in fitted_para.keys()]
np.mean(data), np.std(data)**2

bins = np.linspace(0, 3, 20)  # fixed number of bins
# plt.xlim([min(data)-5, max(data)+5])
plt.hist(data, bins=bins, alpha=0.5)
plt.title('Recovery rate')
plt.xlabel('Fitted rate k')
plt.ylabel('Count')

## plot fitted curve
y_std2fit = np.array(df_month.iloc[9:9 + 12, 66]) # 66 is for King County
(L_, k_, t0_, b_), pcov = opt.curve_fit(f, x_2fit, y_std2fit, p0=[5, 1, 5, -1], maxfev=5000, method="trf")

y_fitted = f(x_2fit, L_, k_, t0_, b_)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plt.xlabel("Date"), plt.ylabel("Standardized Apple data")
ax.plot(x_2fit, y_std2fit, 'o', label='Data')
ax.plot(x_2fit, y_fitted, '-', label="Fitted curve")
# plt.gca().legend(('Data','Fitted curve'))
ax.legend(loc="upper left")
plt.show()

"""
    rescale 
"""
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\King_KCM_ST_transit_monthly.csv')
data = agencymonthly.iloc[3:,]['KCM_ST'].values.reshape(-1,1)
## standardization
df_data = pd.DataFrame(data)
mean_data = np.array(df_data.mean(axis=0))
std_data = np.array(df_data.std(axis=0))

df_data = df_data.subtract(mean_data, axis=1)
df_data = df_data.div(std_data, axis=1)

sys.path.extend(['C:\\Users\\wangf\\Python\\covid_w\\compareBigAndSmall\\bayesian_changepoint_detection-master'])
from bayesian_changepoint_detection.generate_data import generate_normal_time_series
from bayesian_changepoint_detection.priors import const_prior
from functools import partial
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
import bayesian_changepoint_detection.offline_likelihoods as offline_ll

data = df_data.values.reshape(-1,1)

prior_function = partial(const_prior, p=1/(len(data) + 1))

Q, P, Pcp = offline_changepoint_detection(data, prior_function ,offline_ll.StudentT(), truncate=-40)

len(Pcp)
for p_i in np.exp(Pcp).sum(0): print(p_i)

fig, ax = plt.subplots(2, figsize=[8, 10], sharex=True)
ax[0].plot(data[:])
ax[1].plot(np.exp(Pcp).sum(0))
plt.title('Agency - LAC')
plt.show()

"""
   processing Census 2020 data
"""
censusData = []
## ------------- Occupied
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.H3-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['H3_002N'] = census['H3_002N'].astype(float) / census['H3_001N'].astype(float) * 100

census = census.loc[:, ['GEO_ID','NAME', 'H3_002N']]
censusData.append(census.copy())
## ------- Renter occupied (others: Owned with a mortgage or a loan)
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.H4A-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['H4A_004N'] = census['H4A_004N'].astype(float) / census['H4A_001N'].astype(float) * 100

census = census.loc[:, ['GEO_ID', 'H4A_004N']]
censusData.append(census.copy())

## ------- Householder who is White alone; Householder who is Black or African American alone
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.H6-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['H6_002N'] = census['H6_002N'].astype(float) / census['H6_001N'].astype(float) * 100 # white
census['H6_003N'] = census['H6_003N'].astype(float) / census['H6_001N'].astype(float) * 100 # black
census = census.loc[:, ['GEO_ID', 'H6_002N', 'H6_003N']]
censusData.append(census.copy())

## ------- Owner occupied:!!1-person household
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.H12A-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['H12A_003N'] = census['H12A_003N'].astype(float) / census['H12A_001N'].astype(float) * 100 # Owner occupied:!!1-person household
census['H12A_005N'] = census['H12A_005N'].astype(float) / census['H12A_001N'].astype(float) * 100 # Owner occupied:!!3-person household
census['H12A_011N'] = census['H12A_011N'].astype(float) / census['H12A_001N'].astype(float) * 100 # Renter occupied:!!1-person household
census['H12A_013N'] = census['H12A_013N'].astype(float) / census['H12A_001N'].astype(float) * 100 # Renter occupied:!!3-person household

census = census.loc[:, ['GEO_ID', 'H12A_003N', 'H12A_005N', 'H12A_011N', 'H12A_013N']]
censusData.append(census.copy())

## ------- Owner/Renter occupied:!!Householder 15 to 44 years"
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.H13A-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['H13A_003N'] = (census['H13A_003N'].astype(float)+census['H13A_004N'].astype(float)+census['H13A_005N'].astype(float)) / census['H13A_002N'].astype(float) * 100 # Owner occupied:!!Householder 15 to 44 years"
census['H13A_013N'] = (census['H13A_013N'].astype(float)+census['H13A_014N'].astype(float)+census['H13A_015N'].astype(float)) / census['H13A_012N'].astype(float) * 100 # Renter occupied:!!Householder 15 to 44 years"

census = census.loc[:, ['GEO_ID', 'H13A_003N', 'H13A_013N']]
censusData.append(census.copy())

## ------- Owner occupied:!!With children under 18 years
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.H15-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['H15_003N'] = census['H15_003N'].astype(float) / census['H15_002N'].astype(float) * 100 # white
census['H15_006N'] = census['H15_006N'].astype(float) / census['H15_005N'].astype(float) * 100 # black
census = census.loc[:, ['GEO_ID', 'H15_003N', 'H15_006N']]
censusData.append(census.copy())

## ------- Urban
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.P2-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['P2_002N'] = census['P2_002N'].astype(float) / census['P2_001N'].astype(float) * 100 # urban
census = census.loc[:, ['GEO_ID', 'P2_002N']]
censusData.append(census.copy())

## ------- Race
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.P3-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['P3_002N'] = census['P3_002N'].astype(float) / census['P3_001N'].astype(float) * 100 # White
census['P3_003N'] = census['P3_003N'].astype(float) / census['P3_001N'].astype(float) * 100 # White
census = census.loc[:, ['GEO_ID', 'P3_002N', 'P3_003N']]
censusData.append(census.copy())

## ------- POPULATION IN HOUSEHOLDS BY AGE
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.P15-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['P15_002N'] = census['P15_002N'].astype(float) / census['P15_001N'].astype(float) * 100 # Under 18 years
census = census.loc[:, ['GEO_ID', 'P15_002N']]
censusData.append(census.copy())

## ------- 2-or-more-person household:!!Family households:!!Married couple family:!!With own children under 18 years
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.PCT2-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['PCT2_002N'] = census['PCT2_002N'].astype(float) / census['PCT2_001N'].astype(float) * 100 # 1-person household
census['PCT2_008N'] = census['PCT2_008N'].astype(float) / census['PCT2_001N'].astype(float) * 100 # 2-or-more-person household:!!Family households:!!Married couple family:!!With own children under 18 years
census = census.loc[:, ['GEO_ID', 'PCT2_002N', 'PCT2_008N']]
censusData.append(census.copy())

## ------- Married couple family:!!With own children under 18 years:
census = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\census2020\{}.csv'.format("DECENNIALDHC2020.PCT10-Data"),
    encoding='latin-1')
census.drop(index=census.index[0], axis=0, inplace=True)

census['PCT10_003N'] = census['PCT10_003N'].astype(float) / census['PCT10_002N'].astype(float) * 100 # Married couple family:!!With own children under 18 years
census['PCT10_004N'] = census['PCT10_004N'].astype(float) / census['PCT10_003N'].astype(float) * 100 # Married couple family:!!With own children under 18 years:!!Under 6 years only

census = census.loc[:, ['GEO_ID', 'PCT10_003N', 'PCT10_004N']]
# census['GEO_ID'] = census['GEO_ID'].apply(lambda x: x[9:]).astype(int)  # .astype(str)
# census.set_index('GEO_ID', inplace=True)
censusData.append(census.copy())


# """
# process ACSDP5Y2019 "mata" data
# """
# ACSDP = pd.read_csv(
#     r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\ACSDP5Y2019\ACSDP5Y2019.DP03-Column-Metadata.csv',
#     encoding='latin-1')
# ACSDP.drop(index=ACSDP.index[0], axis=0, inplace=True)
# ACSDP.drop(index=ACSDP.index[0], axis=0, inplace=True)
# ACSDP.drop(index=ACSDP.index[0], axis=0, inplace=True)
#
# ACSDP = ACSDP.loc[list(map(lambda x: x.endswith('E'), ACSDP['Column Name']))]
# ACSDP.to_csv(
#     r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\ACSDP5Y2019\ACSDP5Y2019.DP03-Column-Metadata-copy1.csv',
#     index=False)

"""
    Socio-demo information
"""
# DP03_0001E: Estimate!!EMPLOYMENT STATUS!!Population 16 years and over
# DP03_0021E: Estimate!!COMMUTING TO WORK!!Workers 16 years and over!!Public transportation (excluding taxicab)
# "DP03_0119E","Estimate!!PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW THE POVERTY LEVEL!!All families"

ACSDP = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\ACSDP5Y2019\ACSDP5Y2019.DP03-Data.csv',
    encoding='latin-1')
ACSDP.drop(index=ACSDP.index[0], axis=0, inplace=True)
ACSDP['perc_transit_commute'] = ACSDP['DP03_0021E'].astype(float) / ACSDP['DP03_0001E'].astype(float) * 100
# ACSDP['DP03_0119E'] = ACSDP['DP03_0119E'].astype(float)
ACSDP['GEO_ID'] = ACSDP['GEO_ID'].apply(lambda x: x[9:]).astype(int)  # .astype(str)
ACSDP = ACSDP.loc[:, ['GEO_ID', 'perc_transit_commute']]
ACSDP.set_index('GEO_ID', inplace=True)
# ACSDP.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\ACSDP5Y2019\percent_commuting_pub_transit.csv',index=True)

jn = ACSDP
for census in censusData:
    census['GEO_ID'] = census['GEO_ID'].apply(lambda x: x[9:]).astype(int)  # .astype(str)
    census.set_index('GEO_ID', inplace=True)

    jn = pd.merge(jn, census, left_index=True, right_index=True)


"""
   process land use data
"""
# POPDEN_COU	2020 population density of the County (square miles)
# POPPCT_URB	Percent of the 2020 Census population of the County within Urban blocks
# ALAND_PCT_URB	Percent of 2020 land within the County that is classified as Urban
# HOUPCT_RUR	Percent of the 2020 housing units in the County within Rural blocks
# ALAND_PCT_RUR	Percent of 2020 land within the County that is classified as Rural
landuse = pd.read_excel(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\2020_UA_COUNTY.xlsx', sheet_name='2020_UA_COUNTY', dtype = {'STATE': str, 'COUNTY': str})
    # index_col=0, skiprows=[0, 1, 2, 3])  #
landuse.columns
landuse['GEO_ID'] = landuse['STATE'] + landuse['COUNTY']
landuse['GEO_ID'] = landuse['GEO_ID'].astype(int)
landuse = landuse.loc[:, ['GEO_ID', 'POPDEN_COU', 'POPPCT_URB', 'ALAND_PCT_URB', 'HOUPCT_RUR', 'ALAND_PCT_RUR']]
landuse.set_index('GEO_ID', inplace=True)

jn = pd.merge(jn, landuse, left_index=True, right_index=True)


##
PopulationEstimates = pd.read_excel(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\PopulationEstimates.xlsx',
    index_col=0, skiprows=[0, 1, 2, 3])  #
PopulationEstimates.columns
# # PopulationEstimates["Federal Information Processing Standards (FIPS) Code"] = PopulationEstimates["Federal Information Processing Standards (FIPS) Code"].astype(str)
# # PopulationEstimates.set_index("Federal Information Processing Standards (FIPS) Code", inplace=True)
# PopulationEstimates = PopulationEstimates.drop(['Rural-Urban Continuum Code 2013', 'Population 1990',
#                                                 'Population 2000', 'Population 2010', 'Population 2021'], axis=1)
PopulationEstimates = PopulationEstimates.loc[:, ['Population 2020']]
jn = pd.merge(jn, PopulationEstimates, left_index=True, right_index=True)
##
PovertyEstimates = pd.read_excel(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\PovertyEstimates.xlsx',
                                 index_col=0, skiprows=[0, 1, 2, 3])  #
PovertyEstimates.columns
PovertyEstimates = PovertyEstimates.loc[:, ['PCTPOVALL_2020']] # 'Stabr', 'Area_name',

jn = pd.merge(jn, PovertyEstimates, left_index=True, right_index=True)
##
Unemployment = pd.read_excel(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\Unemployment.xlsx',
                             index_col=0, skiprows=[0, 1, 2, 3])
Unemployment.columns
Unemployment = Unemployment.loc[:, ['Unemployment_rate_2021']] # 'State', 'Area_name',

jn = pd.merge(jn, Unemployment, left_index=True, right_index=True)
##
Education = pd.read_excel(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\Education.xlsx',
                          index_col=0, skiprows=[0, 1, 2])
Education.columns
Education = Education.loc[:, ["Percent of adults with a bachelor's degree or higher, 2017-21"]] #'State', 'Area name',

jn = pd.merge(jn, Education, left_index=True, right_index=True)


"""
   process covid data and mask use
"""
covidcase = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\covid-19-data_us-counties-2021.csv',dtype = {'fips': str},
    keep_default_na=False,
    encoding='latin-1')
covidcase.dtypes
covidcase = covidcase.loc[:, ['fips', 'cases', 'deaths']]
covidcase.columns = ['GEO_ID', 'cases', 'deaths']
covidcase = covidcase.loc[list(map(lambda x: len(x)>0, covidcase['GEO_ID'])), ['GEO_ID', 'cases', 'deaths']] # some GEO_ID is empty
covidcase['GEO_ID'] = covidcase['GEO_ID'].astype(int) # list(covidcase['GEO_ID'])
covidcase = covidcase.groupby(['GEO_ID'])['cases', 'deaths'].max()
# covidcase.set_index('GEO_ID', inplace=True)

# jn = jn.drop(['cases_x', 'cases_y', 'cases', 'deaths_x', 'deaths_y'], axis=1)
jn = pd.merge(jn, covidcase, left_index=True, right_index=True)
jn['cases'] = jn['cases'] / jn['Population 2020'] * 10000
jn['deaths'] = jn['deaths'].astype(float) / jn['Population 2020'] * 10000

maskuse = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\mask-use-by-county.csv', dtype = {'COUNTYFP': int},
    encoding='latin-1')
maskuse.dtypes
maskuse = maskuse.loc[:, ['COUNTYFP', 'ALWAYS']]
maskuse.columns = ['GEO_ID', 'alwaysUseMask']
maskuse.set_index('GEO_ID', inplace=True)
jn = pd.merge(jn, maskuse, left_index=True, right_index=True)


"""
    join SVI data
"""
SVI = pd.read_csv(
    r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\SVI_2020_US_county.csv',
    encoding='latin-1')
SVI['FIPS'] = SVI['FIPS'].astype(int)
SVI.set_index('FIPS', inplace=True)
SVI = SVI.iloc[:,6:]

""" 
   processing
"""
# jn.columns
# jn = jn.loc[:, ['NAME', 'perc_transit_commute', 'Population 2020', 'PCTPOVALL_2020', 'Unemployment_rate_2021',
#                 "Percent of adults with a bachelor's degree or higher, 2017-21"]]
jn['Population 2020'] = jn['Population 2020'] / 1000000
# jn.columns = ['county', 'transit', 'population', 'poverty', 'unemploy', 'education']

with open(r"C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\joint0704.pkl", 'wb') as fp:
    pickle.dump(jn, fp)

with open(r"C:\Users\wangf\Python\covid_w\compareBigAndSmall\socio_demo\joint0704.pkl", 'rb') as f:
    jn = pickle.load(f)

jn = pd.merge(jn, SVI, left_index=True, right_index=True)

jn.set_index("NAME", inplace=True)

"""
     linear regression
"""
with open(r"C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\Apple_fitted_para_R0.7.pkl", 'rb') as f:
    fitted_para = pickle.load(f)
df_para = pd.DataFrame.from_dict(fitted_para, orient='index')
df_para.columns = ['L', 'k', 't0', 'b']
df_para.shape
df_para = df_para[df_para.t0 < 12]
df_para = df_para[df_para.k < 4]
df_para = df_para[df_para.L < 500]

model_data = pd.merge(df_para, jn, left_index=True, right_index=True)
model_data.shape

corr_matrix = model_data.corr()


from scipy.stats import gaussian_kde

def plot_scatter_density(x, y):
    # Calculate the point density
    # x, y = model_data.iloc[:,col].tolist(), model_data.iloc[:,1].tolist()
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = list(z.argsort())
    # x, y, z = x[idx], y[idx], z[idx]
    x, y, z = [x[i] for i in idx], [y[i] for i in idx], [z[i] for i in idx]
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=20, edgecolor='None')
    # plt.colorbar()
    fig.show()

for col in range(4, model_data.shape[1]):
    corrXY = np.corrcoef(model_data.iloc[:,1].tolist(), model_data.iloc[:,col].tolist())[0,1]
    print(corrXY)
    if abs(corrXY) > 0.15:
        plot_scatter_density(model_data.iloc[:,col].tolist(), model_data.iloc[:,1].tolist())


### model X Y
Y = model_data.iloc[:, 1]  # .values.reshape(-1, 1)  # values converts it into a numpy array
X = model_data.iloc[:, 4:]  # .values # -1 means that calculate the dimension of rows, but have 1 column

X['H12A_005N'] = X['H12A_003N'] + X['H12A_005N']
X = X.drop(['Population 2020', 'H4A_004N','H12A_003N','H12A_011N','H12A_013N', 'EPL_MOBILE', 'E_CROWD', 'M_HU', 'EPL_AGE65', 'M_NHPI'], axis = 1)
X = X.drop(['M_GROUPQ'], axis = 1)

X['E_HISP'] = np.log10(X['E_HISP'].values)
X['E_NHPI'] = np.log10(X['E_NHPI'].values)
X['EP_MOBILE'] = np.log10(X['EP_MOBILE'].values)

# from sklearn.linear_model import LinearRegression
# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(X, Y)  # perform linear regression
# linear_regressor.score(X, Y)


import statsmodels.api as sm

X = sm.add_constant(X)

results = sm.OLS(Y, X).fit()
results.summary()
X.shape


for county_i in list(fitted_para.keys()):
    print(county_i,'\t', fitted_para[county_i][1])

"""
    feature importance
"""
from sklearn.ensemble import RandomForestRegressor
# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

# y = df_data.loc[:,['y']]
# X = df_data.drop(['y', 'ID'], axis=1)
# print(X.shape, X.columns)

# X = sm.add_constant(X)

## scale or not ??
scaler = StandardScaler().fit(X)
Xsc = scaler.transform(X)
Xsc = pd.DataFrame(data=Xsc, columns=X.columns)

Xsc['const'] = 1.
# Xsc.rename(dict(zip([range(len(Xsc.shape[1])), X.columns])), axis='columns')# dict(zip([0,1],['a','b']))
# Xsc = sm.add_constant(Xsc) ## add an intercept (beta_0) to our model
# ## Split data to train and test on 80-20 ratio
# Xtrain_rf, Xtest_rf, ytrain_rf, ytest_rf = train_test_split(Xsc, y, test_size = 0.2, random_state = 42)
# len(Xtest)

Xsc = pd.DataFrame(data=X, columns=X.columns)

rf = RandomForestRegressor(n_estimators=1000, n_jobs=5) #, max_depth=8
rf.fit(Xsc, Y)
## R-squared: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.score
acc = rf.score(Xsc, Y) # Return the coefficient of determination of the prediction.
print("Accuracy: {}%".format(acc * 100))
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(list(enumerate(Xsc.columns)), importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
for feature in feature_importances:
    print ('{}\t{}'.format(feature[0],feature[1]))

Xsc_sorted = Xsc[[feature[0][1] for feature in feature_importances]]



## vis importance of features
xlables = ['PopDen', 'Family Child', 'Owner Occ', 'Mobile House', 'Occupied', 'Age65', 'Land Rural', 'CovidDeath', 'HISP', 'NHPI'] # Native Hawaiian or Other Pacific Islander (NHPI)
plt.figure(figsize=(7, 6))
plt.title("Feature Importances")
plt.bar(range(len(feature_importances[:10])), [feature[1] for feature in feature_importances[:10]], align="center")
# plt.xticks(range(len(feature_importances[:20])), [feature[0] for feature in feature_importances[:20]], rotation=90)
plt.xticks(range(len(feature_importances[:10])), xlables, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()

## compare with predicted values
y_predicted = rf.predict(Xsc).tolist()
# plot_scatter_density(Y.tolist(), rf.predict(Xsc).tolist())
fig = plt.figure()
plt.scatter(Y.tolist(), y_predicted, alpha=0.7, c = 'blue', s=20, edgecolor='None')
plt.xlim(0.4, 3.5), plt.ylim(0.4, 3.5)
plt.xlabel('Recovery rate from data'), plt.ylabel('Predicted recovery rate')

## plot partial plots
from sklearn.inspection import partial_dependence, plot_partial_dependence
## https://www.kaggle.com/dansbecker/partial-dependence-plots
Xsc.shape
my_plots = plot_partial_dependence(rf,
                                   X=Xsc,  # raw predictors data.
                                   features=[feature[0][0] for feature in feature_importances[:10]],  # column numbers of plots we want to show
                                   n_jobs=5,
                                   grid_resolution=10)  # number of values to plot on x axis
plt.tight_layout()


## feature selection
Xsc.shape
model = SelectFromModel(rf, prefit=True, max_features=10)
X_select = model.transform(Xsc)
X_select.shape

rf_select = RandomForestRegressor(n_estimators=1000, n_jobs=5) #, max_depth=8
rf_select.fit(X_select, Y)
## R-squared: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.score
acc_select = rf_select.score(X_select, Y) # Return the coefficient of determination of the prediction.
print("Accuracy: {}%".format(acc_select * 100))

my_plots = plot_partial_dependence(rf_select,
                                   X=X_select,  # raw predictors data.
                                   features=list(range(X_select.shape[1])),  # column numbers of plots we want to show
                                   n_cols=2, feature_names = xlables,
                                   n_jobs=5,
                                   grid_resolution=10)  # number of values to plot on x axis
plt.subplots_adjust(top=0.9)
# set vetical gap: 0.4