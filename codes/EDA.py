
"""
    April 21, 2021
"""
"""
## google data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys, os

# os.chdir('/media/Data/U_ProgramData/Python/Covid-19/googleData')
# os.chdir('C:\Users\wangf\Python\Covid-19\googleData')


url = "https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/google_reports/mobility_report_US.csv"
rawdata = pd.read_csv(url) #mobility_report_US.csv
df_WA = rawdata.loc[rawdata['state'] == "Washington"]

# King County, Kitsap County, Pierce County, and Snohomish County
df = df_WA.loc[df_WA['county'].isin(['King County', 'Kitsap County', 'Pierce County', 'Snohomish County']) ]#"Pierce County"

# df.drop(['state', 'county'], axis=1, inplace=True)

df.dtypes
df['date'] = pd.to_datetime(df["date"])#.dt.date

df.set_index('date', inplace=True)
df.columns

# ##
# df_weekly = df.resample("W").sum()/7
# df_weekly.to_csv("googleMob_Pierce.csv")

# 'retail and recreation', 'grocery and pharmacy',
#        'parks', 'transit stations', 'workplaces', 'residential'

catogary = 'transit stations' #'workplaces'#
df_catogary = df.loc[df['county'].isin(['King County'])][[catogary]]
for county in ['Kitsap County', 'Pierce County', 'Snohomish County']:
    df_catogary[county] = df.loc[df['county']==county][[catogary]].values
df_catogary.columns = ['King County', 'Kitsap County', 'Pierce County', 'Snohomish County']

df_catogary_wd_daily = df_catogary[df_catogary.index.weekday<=4] # monday is day 0
df_catogary_wd_daily.plot(grid=True, ylabel='Percent change (% daily)', title=catogary+' (Weekdays)')

df_catogary_wd = df_catogary[df_catogary.index.weekday<=4].resample("W").sum()/5 # monday is day 0
df_catogary_wd.plot(grid=True, ylabel='Percent change', title=catogary+' (Weekdays)', ylim=(-80,45))
df_catogary_wd.show()

df_catogary_end = df_catogary[df_catogary.index.weekday>4].resample("W").sum()/2 # monday is day 0
df_catogary_end.plot(grid=True, ylabel='Percent change', title=catogary+' (Weekends)', ylim=(-80,45))


df_catogary_weekly = df_catogary.resample("W").sum()/7
df_catogary = df_catogary_weekly

df_catogary_monthly = df_catogary.resample("M").mean()

# df.plot()

plt.plot(df['retail and recreation'])

plt.plot(df['transit stations'])

df.to_csv("googleMob_daily_Snohomish.csv")

########## --- downloaded data from: https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip
# combine three years' data
rawdata2020 = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_US_Region_Mobility_Report.csv')
rawdata2021 = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2021_US_Region_Mobility_Report.csv')
rawdata2022 = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2022_US_Region_Mobility_Report.csv')
rawdata2020_2022 = pd.concat([rawdata2020,rawdata2021,rawdata2022])
rawdata2020_2022.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report.csv',index=False)

rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report.csv')

df_WA = rawdata.loc[rawdata['sub_region_1'] == "Washington"]
df = df_WA.loc[df_WA['sub_region_2'].isin(['King County']) ]#, 'Kitsap County', 'Pierce County', 'Snohomish County'

df.dtypes
df['date'] = pd.to_datetime(df["date"])#.dt.date

df.set_index('date', inplace=True)
df.columns

catogary = 'transit_stations_percent_change_from_baseline' #'workplaces'#
df_catogary = df[[catogary]]

df_catogary_monthly = df_catogary.resample("M").mean()

## New York City
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report.csv')
df_NY = rawdata.loc[rawdata['sub_region_1'] == "New York"]
df1 = df_NY.loc[df_NY['sub_region_2'].isin(['Bronx County']) ][['date','transit_stations_percent_change_from_baseline']]
df2 = df_NY.loc[df_NY['sub_region_2'].isin(['Kings County']) ][['date','transit_stations_percent_change_from_baseline']]
df3 = df_NY.loc[df_NY['sub_region_2'].isin(['New York County']) ][['date','transit_stations_percent_change_from_baseline']]
df4 = df_NY.loc[df_NY['sub_region_2'].isin(['Queens County']) ][['date','transit_stations_percent_change_from_baseline']]
df5 = df_NY.loc[df_NY['sub_region_2'].isin(['Richmond County']) ][['date','transit_stations_percent_change_from_baseline']]

# population of the 5 counties: np.array([1.427, 2.577, 1.629, 2.271, 0.4756]).sum() = 8.3796
df = df1.copy()
df['transit_stations_percent_change_from_baseline'] = \
    (df1['transit_stations_percent_change_from_baseline'].values * 1.427 +
     df2['transit_stations_percent_change_from_baseline'].values * 2.577 +
     df3['transit_stations_percent_change_from_baseline'].values * 1.629 +
     df4['transit_stations_percent_change_from_baseline'].values * 2.271 +
     df5['transit_stations_percent_change_from_baseline'].values * 0.4756)/8.3796

df['date'] = pd.to_datetime(df["date"])#.dt.date
df.set_index('date', inplace=True)

# df = df.groupby('date')[['transit_stations_percent_change_from_baseline']].mean()

df_monthly = df_catogary.resample("M").mean()

## Washington Metropolitan Area
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report.csv')
countiesInWMA = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\counties_WMA_wiki.csv')
countiesInWMA.shape # (23, 3)
# take the first county out as initiation
countiesTrends = rawdata.loc[rawdata['sub_region_1']=='Virginia'].loc[rawdata['sub_region_2']=='Fairfax County'][['date','transit_stations_percent_change_from_baseline']]
countiesTrends['transit_stations_percent_change_from_baseline'] = 1147532.0 * countiesTrends['transit_stations_percent_change_from_baseline'].values
for i in range(1,countiesInWMA.shape[0]):
    place = countiesInWMA.iloc[i].values
    if place[0] == 'District of Columbia':
        trend_i = rawdata.loc[rawdata['sub_region_1'] == "District of Columbia"][['date','transit_stations_percent_change_from_baseline']]
    else:
        trend_i = rawdata.loc[rawdata['sub_region_1']==place[0]].loc[rawdata['sub_region_2']==place[1]][['date','transit_stations_percent_change_from_baseline']]
    trend_i.columns = ['date', place[1]]
    trend_i[place[1]] = place[2] * trend_i[place[1]].values # place[2] gives # of population
    countiesTrends = pd.merge(countiesTrends, trend_i, on='date')
# use the first 8 counties only, other counties (23-8) have many missing values
countiesTrends['mean'] = countiesTrends.iloc[:,:8].sum(axis=1)/sum(countiesInWMA['population (2019 Estimate)'].values[:8])
countiesTrends = countiesTrends[['date','mean']]

countiesTrends['date'] = pd.to_datetime(countiesTrends["date"])#.dt.date
countiesTrends.set_index('date', inplace=True)

countiesTrends_monthly = countiesTrends.resample("M").mean()

# df = df.groupby('date')[['transit_stations_percent_change_from_baseline']].mean()

df_monthly = df_catogary.resample("M").mean()

# countiesNames = list(zip(countiesInWMA['State'].values, countiesInWMA['County'].values))
# for place in countiesNames:
#     if place[0] == 'District of Columbia':
#         countiesTrends.append(rawdata.loc[rawdata['sub_region_1'] == "District of Columbia"]['transit_stations_percent_change_from_baseline'].values)
#     else:
#         countiesTrends.append(rawdata.loc[rawdata['sub_region_1']==place[0]].loc[rawdata['sub_region_2']==place[1]]['transit_stations_percent_change_from_baseline'].values)
# len(countiesTrends)
# len(countiesInWMA['population (2019 Estimate)'].values)
# sum_trends = np.sum([countiesTrends[i]*countiesInWMA['population (2019 Estimate)'].values[i] for i in range(len(countiesTrends))])
#
# for i in range(len(countiesTrends)):
#     print(len(countiesTrends[i]))

## LA county
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report.csv')
countiesTrends = rawdata.loc[rawdata['sub_region_1']=='California'].loc[rawdata['sub_region_2']=='Los Angeles County'][['date','transit_stations_percent_change_from_baseline']]
countiesTrends.columns = ['date','transit_google']

countiesTrends['date'] = pd.to_datetime(countiesTrends["date"])#.dt.date
countiesTrends.set_index('date', inplace=True)

df_monthly = countiesTrends.resample("M").mean()

countiesTrends.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report-LAC-daily.csv') # , index=False
df_monthly.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report-LAC-monthly.csv')

"""
    ### start here. -->> compare: King -----------------------------------------------
"""

# 2020_2022_US_Region_Mobility_Report-king-monthly, 2020_2022_US_Region_Mobility_Report-DC-monthly, 2020_2022_US_Region_Mobility_Report-LAC-monthly, 2020_2022_US_Region_Mobility_Report-NYC-monthly
# applemobilitytrends_king_transit_monthly_change, applemobilitytrends_LAC_transit_monthly_change, applemobilitytrends_NYC_transit_monthly_change, applemobilitytrends_WMA_transit_monthly_change
# King_KCM_ST_transit_monthly, NYC_transit_monthly, WashingtonMetropolitanAreaTransitAuthority_monthly_change,
googlemonthly_king = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report-king-monthly.csv')
applemonthly_king = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_king_transit_monthly_change.csv')
agencymonthly_king = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\King_KCM_ST_transit_monthly.csv')

googlemonthly_king.dtypes
googlemonthly_king['date'] = pd.to_datetime(googlemonthly_king["date"]).dt.strftime('%m/%Y')
googlemonthly_king.columns = ['date', 'transit_google']
applemonthly_king['date'] = pd.to_datetime(applemonthly_king["date"]).dt.strftime('%m/%Y')
applemonthly_king.columns = ['date', 'transit_apple']
agencymonthly_king['date'] = pd.to_datetime(agencymonthly_king["date"]).dt.strftime('%m/%Y')
agencymonthly_king.columns = ['date', 'transit_agency']
# googlemonthly_seattle.set_index('date', inplace=True)
# googlemonthly_seattle.columns

# general comparison
googleApple = pd.merge(googlemonthly_king, applemonthly_king, on='date')
googleAppleAgency = pd.merge(googleApple, agencymonthly_king, on='date')
googleAppleAgency = googleAppleAgency.iloc[2:]
google_sub = googleAppleAgency['transit_google'].values
apple_sub = googleAppleAgency['transit_apple'].values
agency_sub = googleAppleAgency['transit_agency'].values

np.mean(google_sub), np.std(google_sub)
np.mean(apple_sub), np.std(apple_sub)
np.mean(agency_sub), np.std(agency_sub)

cumm_deviation_google_mean = [np.mean(google_sub[:i]-agency_sub[:i]) for i in range(1, len(google_sub)+1)]
cumm_deviation_apple_mean = [np.mean(apple_sub[:i]-agency_sub[:i]) for i in range(1, len(google_sub)+1)]
for i in cumm_deviation_google_mean: print(i)
for i in cumm_deviation_apple_mean: print(i)

applemonthly_king.dtypes
## agency vs google
agencyVSgoogle_King = pd.merge(agencymonthly_king, googlemonthly_king,on='date')
agencyVSgoogle_King['monthSince'] = agencyVSgoogle_King.index
agencyVSgoogle_King.set_index('date', inplace=True)

agencyVSgoogle_King = agencyVSgoogle_King.iloc[2:]
m, b = np.polyfit(agencyVSgoogle_King['transit_agency'], agencyVSgoogle_King['transit_google'], 1) #0.95, 17
alphas = np.linspace(0.1, 1, agencyVSgoogle_King.shape[0])
plt.figure(figsize=(4,3))
plt.scatter(agencyVSgoogle_King['transit_agency'], agencyVSgoogle_King['transit_google'], marker='s', alpha=alphas, color='darkred')
plt.xlabel('AD'), plt.ylabel('Google') #, plt.title('King County')
# plt.xlim((-90,40)), plt.ylim((-90,40))
plt.plot(agencyVSgoogle_King['transit_agency'], m*agencyVSgoogle_King['transit_agency']+b, color='red')
plt.tight_layout()
plt.savefig("GoogleScatter_king.svg")
plt.show()

## calculate corr with p-value
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
# https://stackoverflow.com/questions/3949226/calculating-pearson-correlation-and-significance-in-python
from scipy.stats import linregress
ls_google_king = linregress(agencyVSgoogle_King['transit_google'],
                            agencyVSgoogle_King['transit_agency'])
# LinregressResult(slope=0.7573195665869094, intercept=-31.713653140929353, rvalue=0.8462631703278493, pvalue=9.829300619698455e-08, stderr=0.09941355004314921, intercept_stderr=4.67476633404367)
for i in range(6, len(agencyVSgoogle_King)):
    fit_t = linregress(agencyVSgoogle_King['transit_google'].values[:i], agencyVSgoogle_King['transit_agency'].values[:i])
    print(fit_t.intercept, '\t', fit_t.slope)

# from scipy.stats.stats import pearsonr
# pearsonr(agencyVSgoogle_King['KCM_ST'], agencyVSgoogle_King['transit_stations_percent_change_from_baseline'])

## agency vs apple
agencyVSapple_King = pd.merge(agencymonthly_king, applemonthly_king,on='date')
agencyVSapple_King['monthSince'] = agencyVSapple_King.index
agencyVSapple_King.set_index('date', inplace=True)

agencyVSapple_King = agencyVSapple_King.iloc[3:]
m, b = np.polyfit(agencyVSapple_King['transit_agency'], agencyVSapple_King['transit_apple'], 1)
alphas = np.linspace(0.1, 1, agencyVSapple_King.shape[0])
plt.figure(figsize=(4,3))
plt.scatter(agencyVSapple_King['transit_agency'], agencyVSapple_King['transit_apple'], alpha=alphas, color=u'#1f77b4')
plt.xlabel('AD'), plt.ylabel('Apple')#, plt.title('King County')
plt.plot(agencyVSapple_King['transit_agency'], m*agencyVSapple_King['transit_agency']+b, color=u'#1f77b4')
plt.tight_layout()
plt.savefig("AppleScatter_king.svg")
plt.show()

ls_apple_king = linregress(agencyVSapple_King['transit_apple'], agencyVSapple_King['transit_agency'] )
# LinregressResult(slope=0.3140354309632416, intercept=-55.89505759548755, rvalue=0.9854042941562172, pvalue=3.436190534028894e-19, stderr=0.01131194568101645, intercept_stderr=0.5138515586728426)
for i in range(6, len(agencyVSapple_King)):
    fit_t = linregress(agencyVSapple_King['transit_apple'].values[:i], agencyVSapple_King['transit_agency'].values[:i])
    print(fit_t.intercept, '\t', fit_t.slope)

## KS test: perform Kolmogorov-Smirnov test
from scipy.stats import ks_2samp
google_scaled_king = ls_google_king.intercept + ls_google_king.slope* agencyVSgoogle_King['transit_google'].values
for i in google_scaled_king: print(i)
alphas = np.linspace(0.1, 1, agencyVSgoogle_King.shape[0])
plt.scatter(agencyVSgoogle_King['transit_agency'], google_scaled_king, alpha=alphas)
plt.xlabel('Transit Agency'), plt.ylabel('Google_scaled'), plt.title('King County')
plt.show()

ks_2samp(agencyVSgoogle_King['transit_agency'], agencyVSgoogle_King['transit_google']) # before
# KstestResult(statistic=0.6666666666666666, pvalue=2.340322696273753e-05)
ks_2samp(agencyVSgoogle_King['transit_agency'], google_scaled_king) # after
# KstestResult(statistic=0.2916666666666667, pvalue=0.26283384201555077)

apple_scaled_king = ls_apple_king.intercept + ls_apple_king.slope * agencyVSapple_King['transit_apple'].values
for i in apple_scaled_king: print(i)
alphas = np.linspace(0.1, 1, agencyVSapple_King.shape[0])
plt.scatter(agencyVSapple_King['transit_agency'], apple_scaled_king, alpha=alphas)
plt.xlabel('Transit Agency'), plt.ylabel('Apple_scaled'), plt.title('King County')
plt.show()

ks_2samp(agencyVSgoogle_King['transit_agency'], agencyVSapple_King['transit_apple'])
# KstestResult(statistic=0.52, pvalue=0.000942540804363845)
ks_2samp(agencyVSgoogle_King['transit_agency'], apple_scaled_king)
# KstestResult(statistic=0.15, pvalue=0.8852981810605587)



"""
    ------- compare: NYC ------
""""
googlemonthly_NYC = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report-NYC-monthly.csv')
applemonthly_NYC = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_NYC_transit_monthly_change.csv')
agencymonthly_NYC = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\NYC_transit_monthly.csv')

agencymonthly_NYC.dtypes
googlemonthly_NYC['date'] = pd.to_datetime(googlemonthly_NYC["date"]).dt.strftime('%m/%Y')
googlemonthly_NYC.columns = ['date', 'transit_google']
applemonthly_NYC['date'] = pd.to_datetime(applemonthly_NYC["date"]).dt.strftime('%m/%Y')
applemonthly_NYC.columns = ['date', 'transit_apple']
agencymonthly_NYC['date'] = pd.to_datetime(agencymonthly_NYC["date"]).dt.strftime('%m/%Y')
agencymonthly_NYC.columns = ['date', 'transit_agency']
# googlemonthly_seattle.set_index('date', inplace=True)
# googlemonthly_seattle.columns

# general comparison
googleApple = pd.merge(googlemonthly_NYC, applemonthly_NYC, on='date')
googleAppleAgency = pd.merge(googleApple, agencymonthly_NYC, on='date')
googleAppleAgency = googleAppleAgency.iloc[1:]
google_sub = googleAppleAgency['transit_google'].values
apple_sub = googleAppleAgency['transit_apple'].values
agency_sub = googleAppleAgency['transit_agency'].values

np.mean(google_sub), np.std(google_sub)
np.mean(apple_sub), np.std(apple_sub)
np.mean(agency_sub), np.std(agency_sub)

cumm_deviation_google_mean = [np.mean(google_sub[:i]-agency_sub[:i]) for i in range(1, len(google_sub)+1)]
cumm_deviation_apple_mean = [np.mean(apple_sub[:i]-agency_sub[:i]) for i in range(1, len(google_sub)+1)]
for i in cumm_deviation_google_mean: print(i)
for i in cumm_deviation_apple_mean: print(i)


applemonthly_NYC.dtypes
## agency vs google
agencyVSgoogle_NYC = pd.merge(agencymonthly_NYC,googlemonthly_NYC,on='date')
agencyVSgoogle_NYC['monthSince'] = agencyVSgoogle_NYC.index
agencyVSgoogle_NYC.set_index('date', inplace=True)

agencyVSgoogle_NYC = agencyVSgoogle_NYC.iloc[1:-2]
m, b = np.polyfit(agencyVSgoogle_NYC['transit_agency'], agencyVSgoogle_NYC['transit_google'], 1)
alphas = np.linspace(0.1, 1, agencyVSgoogle_NYC.shape[0])
plt.figure(figsize=(4,3))
plt.scatter(agencyVSgoogle_NYC['transit_agency'], agencyVSgoogle_NYC['transit_google'], marker='s', alpha=alphas, color='darkred')
plt.xlabel('AD'), plt.ylabel('Google') #, plt.title('New York City')
plt.plot(agencyVSgoogle_NYC['transit_agency'], m*agencyVSgoogle_NYC['transit_agency']+b, color='red')
plt.tight_layout()
plt.savefig("GoogleScatter_NYC.svg")
plt.show()

ls_google_NYC= linregress(agencyVSgoogle_NYC['transit_google'], agencyVSgoogle_NYC['transit_agency'])
# LinregressResult(slope=2.514503259198468, intercept=85.6743300079046, rvalue=0.9460566517600422, pvalue=9.57188721128921e-13, stderr=0.17956340426823458, intercept_stderr=7.342312946233717)

for i in range(6, len(agencyVSgoogle_NYC)):
    fit_t = linregress(agencyVSgoogle_NYC['transit_google'].values[:i], agencyVSgoogle_NYC['transit_agency'].values[:i])
    print(fit_t.intercept, '\t', fit_t.slope)

## apple vs. agency
agencyVSapple_NYC = pd.merge(agencymonthly_NYC,applemonthly_NYC,on='date')
agencyVSapple_NYC['monthSince'] = agencyVSapple_NYC.index
agencyVSapple_NYC.set_index('date', inplace=True)

agencyVSapple_NYC = agencyVSapple_NYC.iloc[1:]
m, b = np.polyfit(agencyVSapple_NYC['transit_agency'], agencyVSapple_NYC['transit_apple'], 1)
alphas = np.linspace(0.1, 1, agencyVSapple_NYC.shape[0])
plt.figure(figsize=(4,3))
plt.scatter(agencyVSapple_NYC['transit_agency'], agencyVSapple_NYC['transit_apple'], alpha=alphas, color=u'#1f77b4')
plt.xlabel('AD'), plt.ylabel('Apple')#, plt.title('New York City')
plt.plot(agencyVSapple_NYC['transit_agency'], m*agencyVSapple_NYC['transit_agency']+b, color=u'#1f77b4')
plt.tight_layout()
plt.savefig("AppleScatter_NYC.svg")
plt.show()

ls_apple_NYC=linregress(agencyVSapple_NYC['transit_apple'], agencyVSapple_NYC['transit_agency'])
# LinregressResult(slope=0.7259712538740789, intercept=-2.049872822906176, rvalue=0.9737575290412858, pvalue=2.763089663730135e-16, stderr=0.03537968323300528, intercept_stderr=1.3974173404690502)
for i in range(6, len(agencyVSapple_NYC)):
    fit_t = linregress(agencyVSapple_NYC['transit_apple'].values[:i], agencyVSapple_NYC['transit_agency'].values[:i])
    print(fit_t.intercept, '\t', fit_t.slope)

(1.51*1.0033-1.2501-0.26)/3

# KS test
google_scaled_NYC = ls_google_NYC.intercept + ls_google_NYC.slope * agencyVSgoogle_NYC['transit_google'].values
for i in google_scaled_NYC: print(i)

alphas = np.linspace(0.1, 1, agencyVSgoogle_NYC.shape[0])
plt.scatter(agencyVSgoogle_NYC['transit_agency'], google_scaled_NYC, alpha=alphas)
plt.xlabel('Transit Agency'), plt.ylabel('Google_scaled'), plt.title('New York City')
plt.show()

ks_2samp(agencyVSgoogle_NYC['transit_agency'], agencyVSgoogle_NYC['transit_google'])
# KstestResult(statistic=0.68, pvalue=8.494202585197998e-06)
ks_2samp(agencyVSgoogle_NYC['transit_agency'], google_scaled_NYC)
# KstestResult(statistic=0.16, pvalue=0.914993219397903)

apple_scaled_NYC = ls_apple_NYC.intercept + ls_apple_NYC.slope * agencyVSapple_NYC['transit_apple'].values
for i in apple_scaled_NYC: print(i)

alphas = np.linspace(0.1, 1, agencyVSapple_NYC.shape[0])
plt.scatter(agencyVSapple_NYC['transit_agency'], apple_scaled_NYC, alpha=alphas)
plt.xlabel('Transit Agency'), plt.ylabel('Apple_scaled'), plt.title('New York City')
plt.show()

ks_2samp(agencyVSapple_NYC['transit_agency'], agencyVSapple_NYC['transit_apple'])
# KstestResult(statistic=0.348, pvalue=0.12429453839909903)
ks_2samp(agencyVSapple_NYC['transit_agency'], apple_scaled_NYC)
# KstestResult(statistic=0.13, pvalue=0.9923771607128958)

### cross-correlation
# cross corr VS. STW: https://stats.stackexchange.com/questions/422368/dynamic-time-warping-vs-cross-correlation
#Four ways to quantify synchrony between time series data:
# https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

xcov_monthly = [crosscorr(agencyVSgoogle_King['transit_agency'], agencyVSgoogle_King['transit_google'], lag=i) for i in range(-4,5)]
xcov_monthly = [crosscorr(agencyVSapple_King['transit_agency'], agencyVSapple_King['transit_apple'], lag=i) for i in range(-4,5)]

xcov_monthly = [crosscorr(agencyVSgoogle_NYC['transit_agency'], agencyVSgoogle_NYC['transit_google'], lag=i) for i in range(-4,5)]
xcov_monthly = [crosscorr(agencyVSapple_NYC['transit_agency'], agencyVSapple_NYC['transit_apple'], lag=i) for i in range(-4,5)]


## Dynamic time wrapping
# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
# https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd#:~:text=Dynamic%20Time%20Warping%20is%20used,time%20series%20with%20different%20length.&text=How%20to%20do%20that%3F,total%20distance%20of%20each%20component.
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
#------ King: google
distance = dtw.distance(agencyVSgoogle_King['transit_google'], agencyVSgoogle_King['transit_agency'], window=4) # 91.1
distance = dtw.distance(agencyVSgoogle_King['transit_agency'], google_scaled_king, window=4) # 15.34
# the wrapping
path = dtw.warping_path(agencyVSgoogle_King['transit_agency'], google_scaled_king)
dtwvis.plot_warping(agencyVSgoogle_King['transit_agency'], google_scaled_king, path,
                    filename= r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_googleVSagency_King.png')
# keep all warping paths
d, paths = dtw.warping_paths(agencyVSgoogle_King['transit_agency'], google_scaled_king, window=4, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(agencyVSgoogle_King['transit_agency'], google_scaled_king, paths, best_path,
                         filename=r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_googleVSagency_matrix_King.png')
#------ King: apple
distance = dtw.distance(agencyVSapple_King['transit_apple'], agencyVSapple_King['transit_agency'], window=4) # 183.6
distance = dtw.distance(agencyVSapple_King['transit_agency'], apple_scaled_king, window=4) # 8.1
# the wrapping
path = dtw.warping_path(agencyVSapple_King['transit_apple'], agencyVSapple_King['transit_agency'], window=4)
dtwvis.plot_warping(agencyVSapple_King['transit_apple'], agencyVSapple_King['transit_agency'], path,
                    filename= r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_appleVSagency_King_bfScale.png')

path = dtw.warping_path(agencyVSapple_King['transit_agency'], apple_scaled_king)
dtwvis.plot_warping(agencyVSapple_King['transit_agency'], apple_scaled_king, path,
                    filename= r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_appleVSagency_King.png')
# keep all warping paths
d, paths = dtw.warping_paths(agencyVSapple_King['transit_agency'], apple_scaled_king, window=4, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(agencyVSapple_King['transit_agency'], apple_scaled_king, paths, best_path,
                         filename=r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_appleVSagency_matrix_King.png')

#------ NYC: google
distance = dtw.distance(agencyVSgoogle_NYC['transit_google'], agencyVSgoogle_NYC['transit_agency'],window=4) # 166.3
distance = dtw.distance(agencyVSgoogle_NYC['transit_agency'], google_scaled_NYC) # 31.46439027278168
# the wrapping
path = dtw.warping_path(agencyVSgoogle_NYC['transit_agency'], google_scaled_NYC)
dtwvis.plot_warping(agencyVSgoogle_NYC['transit_agency'], google_scaled_NYC, path,
                    filename= r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_googleVSagency_NYC.png')
# keep all warping paths
d, paths = dtw.warping_paths(agencyVSgoogle_NYC['transit_agency'], google_scaled_NYC, window=4, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(agencyVSgoogle_NYC['transit_agency'], google_scaled_NYC, paths, best_path,
                         filename=r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_googleVSagency_matrix_NYC.png')
#------ NYC: apple
distance = dtw.distance(agencyVSapple_NYC['transit_agency'],agencyVSapple_NYC['transit_apple'],  window=1) # 47.2
distance = dtw.distance(agencyVSapple_NYC['transit_agency'], apple_scaled_NYC) # 18.71312124968023
# the wrapping
def plot_warping_single_ax(s1, s2, path, filename=None, fig=None, ax=None):
    """Plot the optimal warping between to sequences.

    :param s1: From sequence.
    :param s2: To sequence.
    :param path: Optimal warping path.
    :param filename: Filename path (optional).
    :param fig: Matplotlib Figure object
    :param ax: Matplotlib axes.Axes object
    :return: Figure, Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.patches import ConnectionPatch

    if fig is None and ax is None:
        fig, ax = plt.subplots(1,1,figsize=(9,5))
    elif fig is None or ax is None:
        raise TypeError(f'The fig and ax arguments need to be both None or both instantiated.')
    ax.plot(s1)
    ax.plot(s2)
    lines = []
    line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.9}
    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        con = ConnectionPatch(xyA=[r_c, s1[r_c]], coordsA=ax.transData,
                              xyB=[c_c, s2[c_c]], coordsB=ax.transData, **line_options)
        lines.append(con)
    for line in lines:
        fig.add_artist(line)
    ax.set_xticks(ax.get_xticks()[::1])
    plt.xticks(rotation=30)
    plt.legend(['Agency', 'Apple'])
    plt.ylabel('Change in transit use (%)')
    plt.grid(axis='x',which='major',linestyle='--', linewidth=0.3)#, grid_alpha=0.7)
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax

path = dtw.warping_path(agencyVSapple_NYC['transit_agency'], agencyVSapple_NYC['transit_apple'], window=2)
plot_warping_single_ax(agencyVSapple_NYC['transit_agency'], agencyVSapple_NYC['transit_apple'], path,
                    filename= r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_appleVSagency_NYC_bfScale1_wind2.pdf')

path = dtw.warping_path(agencyVSapple_NYC['transit_agency'], apple_scaled_NYC)
dtwvis.plot_warping(agencyVSapple_NYC['transit_agency'], apple_scaled_NYC, path,
                    filename= r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_appleVSagency_NYC.png')
# keep all warping paths
d, paths = dtw.warping_paths(agencyVSapple_NYC['transit_agency'], apple_scaled_NYC, window=4, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(agencyVSapple_NYC['transit_agency'], apple_scaled_NYC, paths, best_path,
                         filename=r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_appleVSagency_matrix_NYC.png')


"""
    ----- compare: Washington Metropolitan Area-----
"""
googlemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report-DC-monthly.csv')
applemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_WMA_transit_monthly_change.csv')
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\WashingtonMetropolitanAreaTransitAuthority_monthly_change.csv')

googlemonthly.dtypes
googlemonthly['date'] = pd.to_datetime(googlemonthly["date"]).dt.strftime('%m/%Y')
googlemonthly.columns = ['date', 'transit_google']
applemonthly['date'] = pd.to_datetime(applemonthly["date"]).dt.strftime('%m/%Y')
applemonthly.columns = ['date', 'transit_apple']
agencymonthly['date'] = pd.to_datetime(agencymonthly["date"]).dt.strftime('%m/%Y')
agencymonthly.columns = ['date', 'transit_agency']
# googlemonthly_seattle.set_index('date', inplace=True)
# googlemonthly_seattle.columns

# general comparison
googleApple = pd.merge(googlemonthly, applemonthly, on='date')
googleAppleAgency = pd.merge(googleApple, agencymonthly, on='date')
googleAppleAgency = googleAppleAgency.iloc[2:]
google_sub = googleAppleAgency['transit_google'].values
apple_sub = googleAppleAgency['transit_apple'].values
agency_sub = googleAppleAgency['transit_agency'].values

np.mean(google_sub), np.std(google_sub)
np.mean(apple_sub), np.std(apple_sub)
np.mean(agency_sub), np.std(agency_sub)

cumm_deviation_google_mean = [np.mean(google_sub[:i]-agency_sub[:i]) for i in range(1, len(google_sub)+1)]
cumm_deviation_apple_mean = [np.mean(apple_sub[:i]-agency_sub[:i]) for i in range(1, len(google_sub)+1)]
for i in cumm_deviation_google_mean: print(i)
for i in cumm_deviation_apple_mean: print(i)

## agency vs google
agencyVSgoogle = pd.merge(agencymonthly, googlemonthly,on='date')
agencyVSgoogle['monthSince'] = agencyVSgoogle.index
agencyVSgoogle.set_index('date', inplace=True)


agencyVSgoogle = agencyVSgoogle.iloc[2:]
m, b = np.polyfit(agencyVSgoogle['transit_agency'], agencyVSgoogle['transit_google'], 1)
alphas = np.linspace(0.1, 1, agencyVSgoogle.shape[0])
plt.figure(figsize=(4,3))
plt.scatter(agencyVSgoogle['transit_agency'], agencyVSgoogle['transit_google'], marker='s', alpha=alphas, color='darkred')
plt.xlabel('AD'), plt.ylabel('Google') #,  plt.title('Washington Metropolitan Area')
plt.plot(agencyVSgoogle['transit_agency'], m*agencyVSgoogle['transit_agency']+b, color='red')
plt.tight_layout()
plt.savefig("GoogleScatter_WMA.svg")
plt.show()

## agency vs apple
agencyVSapple = pd.merge(agencymonthly, applemonthly,on='date')
agencyVSapple['monthSince'] = agencyVSapple.index
agencyVSapple.set_index('date', inplace=True)


agencyVSapple = agencyVSapple.iloc[3:]
m, b = np.polyfit(agencyVSapple['transit_agency'], agencyVSapple['transit_apple'], 1)
plt.figure(figsize=(4,3))
alphas = np.linspace(0.1, 1, agencyVSapple.shape[0])
plt.scatter(agencyVSapple['transit_agency'], agencyVSapple['transit_apple'], alpha=alphas, color=u'#1f77b4')
plt.xlabel('AD'), plt.ylabel('Apple')#, plt.title('Washington Metropolitan Area')
plt.plot(agencyVSapple['transit_agency'], m*agencyVSapple['transit_agency']+b, color=u'#1f77b4')
plt.tight_layout()
plt.savefig("AppleScatter_WMA.svg")
plt.show()

ls_google = linregress(agencyVSgoogle['transit_google'], agencyVSgoogle['transit_agency'])
# LinregressResult(slope=2.073, intercept=42.067, rvalue=0.856, pvalue=1.19e-08, stderr=0.2498, intercept_stderr=12.81)
ls_apple=linregress(agencyVSapple['transit_apple'], agencyVSapple['transit_agency'])
# LinregressResult(slope=0.421, intercept=-52.93, rvalue=0.963, pvalue=1.49e-14, stderr=0.024, intercept_stderr=1.018)
# KS test
google_scaled = ls_google.intercept + ls_google.slope * agencyVSgoogle['transit_google'].values

ks_2samp(agencyVSgoogle['transit_agency'], agencyVSgoogle['transit_google'])
# KstestResult(statistic=0.55, pvalue=0.00035)
ks_2samp(agencyVSgoogle['transit_agency'], google_scaled)
# KstestResult(statistic=0.222, pvalue=0.53)

apple_scaled = ls_apple.intercept + ls_apple.slope * agencyVSapple['transit_apple'].values

ks_2samp(agencyVSapple['transit_agency'], agencyVSapple['transit_apple'])
# KstestResult(statistic=0.64, pvalue=3.96e-05)
ks_2samp(agencyVSapple['transit_agency'], apple_scaled)
# KstestResult(statistic=0.2, pvalue=0.71)

## cross-correlation
xcov_monthly = [crosscorr(agencyVSgoogle['transit_agency'], agencyVSgoogle['transit_google'], lag=i) for i in range(-4,5)]
xcov_monthly = [crosscorr(agencyVSapple['transit_agency'], agencyVSapple['transit_apple'], lag=i) for i in range(-4,5)]

## Dynamic time wrapping
#------ with google
distance = dtw.distance(agencyVSgoogle['transit_google'], agencyVSgoogle['transit_agency'],window=2) # 22.15
distance = dtw.distance(agencyVSgoogle['transit_agency'], google_scaled) # 22.15
# the wrapping
path = dtw.warping_path(agencyVSgoogle['transit_agency'], google_scaled)
dtwvis.plot_warping(agencyVSgoogle['transit_agency'], google_scaled, path,
                    filename= r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_googleVSagency_WMA.png')
# keep all warping paths
d, paths = dtw.warping_paths(agencyVSgoogle['transit_agency'], google_scaled, window=4, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(agencyVSgoogle['transit_agency'], google_scaled, paths, best_path,
                         filename=r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_googleVSagency_matrix_WMA.png')
#------ with apple
distance = dtw.distance(agencyVSapple['transit_apple'], agencyVSapple['transit_agency'], window=2) # 14.18
distance = dtw.distance(agencyVSapple['transit_agency'], apple_scaled) # 14.18
# the wrapping
path = dtw.warping_path(agencyVSapple['transit_agency'], apple_scaled)
dtwvis.plot_warping(agencyVSapple['transit_agency'], apple_scaled, path,
                    filename= r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_appleVSagency_WMA.png')
# keep all warping paths
d, paths = dtw.warping_paths(agencyVSapple['transit_agency'], apple_scaled, window=4, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(agencyVSapple['transit_agency'], apple_scaled, paths, best_path,
                         filename=r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_appleVSagency_matrix_WMA.png')

"""
    ----- compare: Los Angeles County-----
"""

googlemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report-LAC-monthly.csv')
applemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_LAC_transit_monthly_change.csv')
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\Los Angeles County_Metro_monthly_from_isotp.metro.net.csv')
agencymonthly = agencymonthly[['date', 'transit_agency']]

googlemonthly.dtypes
googlemonthly['date'] = pd.to_datetime(googlemonthly["date"]).dt.strftime('%m/%Y')
googlemonthly.columns = ['date', 'transit_google']
applemonthly['date'] = pd.to_datetime(applemonthly["date"]).dt.strftime('%m/%Y')
applemonthly.columns = ['date', 'transit_apple']
agencymonthly['date'] = pd.to_datetime(agencymonthly["date"]).dt.strftime('%m/%Y')
agencymonthly.columns = ['date', 'transit_agency']
# googlemonthly_seattle.set_index('date', inplace=True)
# googlemonthly_seattle.columns

# general comparison
googleApple = pd.merge(googlemonthly, applemonthly, on='date')
googleAppleAgency = pd.merge(googleApple, agencymonthly, on='date')
googleAppleAgency = googleAppleAgency.iloc[2:]
google_sub = googleAppleAgency['transit_google'].values
apple_sub = googleAppleAgency['transit_apple'].values
agency_sub = googleAppleAgency['transit_agency'].values

np.mean(google_sub), np.std(google_sub)
np.mean(apple_sub), np.std(apple_sub)
np.mean(agency_sub), np.std(agency_sub)

cumm_deviation_google_mean = [np.mean(google_sub[:i]-agency_sub[:i]) for i in range(1, len(google_sub)+1)]
cumm_deviation_apple_mean = [np.mean(apple_sub[:i]-agency_sub[:i]) for i in range(1, len(google_sub)+1)]
for i in cumm_deviation_google_mean: print(i)
for i in cumm_deviation_apple_mean: print(i)

## agency vs google
agencyVSgoogle = pd.merge(agencymonthly, googlemonthly,on='date')
agencyVSgoogle['monthSince'] = agencyVSgoogle.index
agencyVSgoogle.set_index('date', inplace=True)


agencyVSgoogle = agencyVSgoogle.iloc[2:]
m, b = np.polyfit(agencyVSgoogle['transit_agency'], agencyVSgoogle['transit_google'], 1)
alphas = np.linspace(0.1, 1, agencyVSgoogle.shape[0])
plt.figure(figsize=(4,3))
plt.scatter(agencyVSgoogle['transit_agency'], agencyVSgoogle['transit_google'], marker='s', alpha=alphas, color='darkred')
plt.xlabel('AD'), plt.ylabel('Google') #, plt.title('Los Angeles County')
plt.plot(agencyVSgoogle['transit_agency'], m*agencyVSgoogle['transit_agency']+b, color='red')
plt.tight_layout()
plt.savefig("GoogleScatter_LA.svg")
plt.show()

## agency vs apple
agencyVSapple = pd.merge(agencymonthly, applemonthly,on='date')
agencyVSapple['monthSince'] = agencyVSapple.index
agencyVSapple.set_index('date', inplace=True)

agencyVSapple = agencyVSapple.iloc[3:]
m, b = np.polyfit(agencyVSapple['transit_agency'], agencyVSapple['transit_apple'], 1)
alphas = np.linspace(0.1, 1, agencyVSapple.shape[0])
plt.figure(figsize=(4,3))
sc=plt.scatter(agencyVSapple['transit_agency'], agencyVSapple['transit_apple'], alpha=alphas, color=u'#1f77b4')
# plt.colorbar(sc, orientation='horizontal', cax=plt.gca().inset_axes([0.1,-0.4,0.8,0.05]))
plt.xlabel('AD'), plt.ylabel('Apple')#, plt.title('Los Angeles County')
plt.plot(agencyVSapple['transit_agency'], m*agencyVSapple['transit_agency']+b, color=u'#1f77b4')
plt.tight_layout()
plt.savefig("AppleScatter_LA.svg")
plt.show()

fig, ax = plt.subplots()
im = plt.scatter(agencyVSapple['transit_agency'], agencyVSapple['transit_apple'], alpha=alphas, color=u'#1f77b4')
fig.colorbar(im, ax=ax)
plt.show()


ls_google = linregress(agencyVSgoogle['transit_google'], agencyVSgoogle['transit_agency'])
# LinregressResult(slope=1.6576, intercept=22.19759, rvalue=0.80922, pvalue=5.5829e-07, stderr=0.24564, intercept_stderr=9.5627)
ls_apple=linregress(agencyVSapple['transit_apple'], agencyVSapple['transit_agency'])
# LinregressResult(slope=0.36809, intercept=-29.613, rvalue=0.9431, pvalue=1.7289e-12, stderr=0.02705, intercept_stderr=1.2054)
# KS test
google_scaled = ls_google.intercept + ls_google.slope * agencyVSgoogle['transit_google'].values

ks_2samp(agencyVSgoogle['transit_agency'], agencyVSgoogle['transit_google'])
# KstestResult(statistic=0.3076923076923077, pvalue=0.17202888984679188)
ks_2samp(agencyVSgoogle['transit_agency'], google_scaled)
# KstestResult(statistic=0.23076923076923078, pvalue=0.5009952475373968)

apple_scaled = ls_apple.intercept + ls_apple.slope * agencyVSapple['transit_apple'].values

ks_2samp(agencyVSapple['transit_agency'], agencyVSapple['transit_apple'])
# KstestResult(statistic=0.44, pvalue=0.014838084605848645)
ks_2samp(agencyVSapple['transit_agency'], apple_scaled)
# KstestResult(statistic=0.2, pvalue=0.7102038997076623)

## cross-correlation
xcov_monthly = [crosscorr(agencyVSgoogle['transit_agency'], agencyVSgoogle['transit_google'], lag=i) for i in range(-4,5)] #**
xcov_monthly = [crosscorr(agencyVSapple['transit_agency'], agencyVSapple['transit_apple'], lag=i) for i in range(-4,5)] #**

## Dynamic time wrapping
#------ with google
distance = dtw.distance(agencyVSgoogle['transit_google'], agencyVSgoogle['transit_agency'],window=2) # 33.23
distance = dtw.distance(agencyVSgoogle['transit_agency'], google_scaled) # 23.52
# the wrapping
path = dtw.warping_path(agencyVSgoogle['transit_agency'], google_scaled)
dtwvis.plot_warping(agencyVSgoogle['transit_agency'], google_scaled, path,
                    filename= r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_googleVSagency_LAC.png')
# keep all warping paths
d, paths = dtw.warping_paths(agencyVSgoogle['transit_agency'], google_scaled, window=4, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(agencyVSgoogle['transit_agency'], google_scaled, paths, best_path,
                         filename=r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_googleVSagency_matrix_LAC.png')
#------ with apple
distance = dtw.distance(agencyVSapple['transit_apple'], agencyVSapple['transit_agency'], window=2) # 93.58
distance = dtw.distance(agencyVSapple['transit_agency'], apple_scaled) # 15.41
# the wrapping
path = dtw.warping_path(agencyVSapple['transit_agency'], apple_scaled)
dtwvis.plot_warping(agencyVSapple['transit_agency'], apple_scaled, path,
                    filename= r'D:C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_appleVSagency_LAC.png')
# keep all warping paths
d, paths = dtw.warping_paths(agencyVSapple['transit_agency'], apple_scaled, window=4, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(agencyVSapple['transit_agency'], apple_scaled, paths, best_path,
                         filename=r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\DTW\dtw_vis_appleVSagency_matrix_LAC.png')


"""
## apple data
"""
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.extend(['/media/Data/U_ProgramData/Python/Covid-19/googleData'])

# os.chdir('/media/Data/U_ProgramData/Python/Covid-19/googleData')
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\2020_apple_mobility_report_US.csv')

rawdata = pd.read_excel(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\apple_mobility_report_US.xlsx',sheet_name='2022')


# url = "https://github.com/ActiveConclusion/COVID19_mobility/blob/master/apple_reports/apple_mobility_report_US.csv"
# rawdata = pd.read_csv(url) #mobility_report_US.csv

df_WA = rawdata.loc[rawdata['state'] == "Washington"]

df_WA.dtypes
df_WA['date'] = pd.to_datetime(df_WA["date"])#.dt.date

df_WA.set_index('date', inplace=True)

# King County, Kitsap County, Pierce County, and Snohomish County
df_King = df_WA.loc[df_WA['county_and_city'] == "King County"]

# df_Pierce = df_WA.loc[df_WA['county_and_city'] == "Pierce County"]
# df_Snohomish = df_WA.loc[df_WA['county_and_city'] == "Snohomish County"]
# df_Kitsap = df_WA.loc[df_WA['county_and_city'] == "Kitsap County"]

df_King = df_King[['transit']]
df_King_month = df_King.resample("M").mean()

# df.drop(['state', 'county_and_city', 'geo_type'], axis=1, inplace=True)

##
df_King_weekly = df_King.resample("W").sum()/7
df_King_weekly.columns = ['driving_King','transit_King','walking_King']
df_Pierce_weekly = df_Pierce.resample("W").sum()/7
df_Pierce_weekly.columns = ['driving_Pierce','transit_Pierce','walking_Pierce']
df_Snohomish_weekly = df_Snohomish.resample("W").sum()/7
df_Snohomish_weekly.columns = ['driving_Snohomish','transit_Snohomish','walking_Snohomish']
df_Kitsap_weekly = df_Kitsap.resample("W").sum()/7
df_Kitsap_weekly.columns = ['driving_Kitsap','transit_Kitsap','walking_Kitsap']

# df_weekly.plot()
joint_outcomes = pd.concat([df_King_weekly[['transit_King']], df_Pierce_weekly[['transit_Pierce']],
                            df_Snohomish_weekly[['transit_Snohomish']]], #, df_Kitsap_weekly[['transit_Kitsap']]
                           axis=1)
plt.figure()
joint_outcomes.plot(lw=2, title='Apple Mobility Data', ylabel='Percent Change', layout='tight')
plt.show()


joint_outcomes.to_csv("appleMob_transit.csv")




"""
    bayesian changepoint detection
    # https://github.com/hildensia/bayesian_changepoint_detection/blob/master/Example_Code.ipynb
    from probability to change points: 
    https://stats.stackexchange.com/questions/470473/minimum-posterior-probability-for-bayesian-changepoint-analysis-in-r
"""
sys.path.extend(['C:\\Users\\wangf\\Python\\covid_w\\compareBigAndSmall\\bayesian_changepoint_detection-master'])
from bayesian_changepoint_detection.generate_data import generate_normal_time_series
from bayesian_changepoint_detection.priors import const_prior
from functools import partial
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
import bayesian_changepoint_detection.offline_likelihoods as offline_ll

# compare: King ---------------------
googlemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report-king-monthly.csv')
applemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_king_transit_monthly_change.csv')
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\King_KCM_ST_transit_monthly.csv')
data = googlemonthly.iloc[2:-2,]['transit_stations_percent_change_from_baseline'].values.reshape(-1,1)
data = applemonthly.iloc[3:,]['transit'].values.reshape(-1,1)
data = agencymonthly.iloc[3:,]['KCM_ST'].values.reshape(-1,1)

### ------- compare: NYC ------
googlemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report-NYC-monthly.csv')
applemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_NYC_transit_monthly_change.csv')
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\NYC_transit_monthly.csv')
data = googlemonthly.iloc[2:-2,]['transit_stations_percent_change_from_baseline'].values.reshape(-1,1)
data = applemonthly.iloc[3:,]['transit_mean5county'].values.reshape(-1,1)
data = agencymonthly.iloc[1:-2,]['transit'].values.reshape(-1,1)

### ----- compare: Washington Metropolitan Area-----
googlemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report-DC-monthly.csv')
applemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_WMA_transit_monthly_change.csv')
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\WashingtonMetropolitanAreaTransitAuthority_monthly_change.csv')
data = googlemonthly.iloc[2:-2,]['mean'].values.reshape(-1,1)
data = applemonthly.iloc[3:,]['transit_apple'].values.reshape(-1,1)
data = agencymonthly.iloc[3:-2,]['transit_agency'].values.reshape(-1,1)

## Compare LA County
googlemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\2020_2022_US_Region_Mobility_Report-LAC-monthly.csv')
applemonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\googleApple\applemobilitytrends_LAC_transit_monthly_change.csv')
agencymonthly = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\agencyData\Los Angeles County_Metro_monthly_from_isotp.metro.net.csv')
agencymonthly = agencymonthly[['date', 'transit_agency']]
data = googlemonthly.iloc[2:-2,]['transit_google'].values.reshape(-1,1)
data = applemonthly.iloc[3:,]['transit_apple'].values.reshape(-1,1)
data = agencymonthly.iloc[3:-1,]['transit_agency'].values.reshape(-1,1)

##
mean_data = data.mean(axis=0)
std_data = data.std(axis=0)

data = data - mean_data #df_data.subtract(mean_data, axis=1)
data = data / std_data # df_data.div(std_data, axis=1)

prior_function = partial(const_prior, p=0.01)#p=0.1 # p=1/(len(data) + 1)

Q, P, Pcp = offline_changepoint_detection(data, prior_function ,offline_ll.StudentT(), truncate=-40)

len(Pcp)
for p_i in np.exp(Pcp).sum(0): print(p_i)

fig, ax = plt.subplots(2, figsize=[8, 10], sharex=True)
ax[0].plot(data[:])
ax[1].plot(np.exp(Pcp).sum(0))
plt.title('Agency - LAC')
plt.show()




"""
   Covid cases 
"""
rawdata2020 = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\covidcases\us-counties-2020.csv')
rawdata2021 = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\covidcases\us-counties-2021.csv')
rawdata2022 = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\covidcases\us-counties-2022.csv')
rawdata2020_2022 = pd.concat([rawdata2020,rawdata2021,rawdata2022])
rawdata2020_2022.to_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\covidcases\us-counties-2020-2022.csv',index=False)

## covid in King county
rawdata = pd.read_csv(r'C:\Users\wangf\Python\covid_w\compareBigAndSmall\covidcases\us-counties-2020-2022.csv')

df_WA = rawdata.loc[rawdata['state'] == "Washington"]
df = df_WA.loc[df_WA['county'].isin(['King']) ]#, 'Kitsap County', 'Pierce County', 'Snohomish County'

df.dtypes
df['date'] = pd.to_datetime(df["date"])#.dt.date

df.set_index('date', inplace=True)
df.columns # ['county', 'state', 'fips', 'cases', 'deaths']

df_catogary = df[['cases']]
df_catogary = df_catogary.diff() # the first day is given NaN, which will be fix next.
df_catogary.loc[['2020-02-28'],:] = df[['cases']].values[0]

df_catogary_7 = df_catogary.rolling(window=7).mean() # notice the first 6 is NaN
df_catogary_7.loc[['2020-02-28','2020-02-29','2020-03-01','2020-03-02','2020-03-03','2020-03-04'],:]\
    = df_catogary.loc[['2020-02-28','2020-02-29','2020-03-01','2020-03-02','2020-03-03','2020-03-04'],:].values

df_catogary_monthly = df_catogary.resample("M").mean()



"""
   ## safegraph data
"""
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.extend(['/media/Data/U_ProgramData/Python/Covid-19/googleData'])

os.chdir('/media/Data/U_ProgramData/Python/Covid-19/googleData')

df = pd.read_csv('foot_traffic_safegraph.csv')

df.dtypes
df['Date'] = df['Date'].apply(lambda x: x[2:12])
df['Date'] = pd.to_datetime(df["Date"]) # .dt.date


df.set_index('Date', inplace=True)

df.plot(lw=2, legend=False)

#######################################
import sys
sys.path.extend(['/media/Data/U_ProgramData/Python/Covid-19/NAB'])

# python run.py -d htmcore --detect --optimize --score --normalize


################################ Traffic data
trafficRaw = pd.read_excel('I5speed1mile.xlsx')
trafficRaw.drop(trafficRaw.columns[0], axis=1, inplace=True) # drop the first column
trafficRaw = pd.melt(trafficRaw) # pandas melt all columns
trafficRaw.shape # (173376, 2)

dateindex = pd.date_range(start='1/1/2019', end='4/22/2021', freq='5min')
dateindex = dateindex[:-1] # len(dateindex)/(24*60/5); last one is 00:00 of the last day
dateindex = dateindex[dateindex.dayofweek < 5] # only weekdays
dateindex.shape # (173376,)

trafficRaw['variable'] = dateindex
trafficRaw.columns = ['timestamp', 'value']

trafficRaw.to_csv("I5speed1mile.csv", index=False)

# trafficRaw['value'].plot()

## aggregate to daily
trafficRaw.set_index(["timestamp"], inplace=True)

trafficRaw_daily = trafficRaw.resample("D").mean()

trafficRaw_daily = trafficRaw_daily[trafficRaw_daily.index.dayofweek < 5] # remove weekends, where are NAN

trafficRaw_daily.to_csv("I5speed1mile_daily.csv")

trafficRaw_daily['value'].plot()


################# Try Matrix Profile
# https://stumpy.readthedocs.io/en/latest/Tutorial_Semantic_Segmentation.html
import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

df = pd.read_csv("./NAB/datamy/mydata/I5volume1mile_daily.csv")
df.head()

plt.plot(df['time'], df['value'])
# ax.locator_params(nbins=10, axis='x')
# plt.locator_params(axis='x', nbins=30)

df = df.iloc[:, 1]

m=20
mp = stumpy.stump(df, m=m)

cac, regime_locations = stumpy.fluss(mp[:, 1], L=m, n_regimes=10, excl_factor=1)


fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
axs[0].plot(range(df.shape[0]), df)
for changeLoc in regime_locations: axs[0].axvline(x=changeLoc, linestyle="dashed")

axs[1].plot(range(cac.shape[0]), cac, color='C1')
for changeLoc in regime_locations: axs[1].axvline(x=changeLoc, linestyle="dashed")
plt.show()


