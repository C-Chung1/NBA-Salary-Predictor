import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy.stats import norm
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


reg_stat = pd.read_csv('C:\\Users\\chanb\\Python_Projects\\nba_regstats.csv')
adv_stat = pd.read_csv('C:\\Users\\chanb\\Python_Projects\\NBA_advstat.csv')
player_salary = pd.read_csv('C:\\Users\\chanb\\Python_Projects\\Player_salary.csv')
Player_Position = pd.read_csv('C:\\Users\\chanb\\Python_Projects\\Player_Position.csv')
current_reg_stat = pd.read_csv('C:\\Users\\chanb\\Python_Projects\\nba_regstats_22.csv')
current_adv_stat = pd.read_csv('C:\\Users\\chanb\\Python_Projects\\NBA_advstat_22.csv')
adv_stat.columns

# dataset containing reg and adv stats but no salary
reg_stat = reg_stat.drop('Rk', axis=1)
adv_stat = adv_stat.drop('Rk', axis=1)
current_reg_stat = current_reg_stat.drop('Rk', axis=1)
current_adv_stat = current_adv_stat.drop('Rk', axis=1)

reg_stat_pos = pd.merge(reg_stat,Player_Position, how='left', left_on=['Player', 'Season'], right_on=['Player', 'Season'])
reg_stat_pos.isnull().sum()
# reg_stat_pos[reg_stat_pos['Pos'].isnull()].index.tolist()
reg_adv = pd.merge(reg_stat_pos,adv_stat, how='left', left_on=['Player', 'Season'], right_on=['Player', 'Season'])
reg_adv.columns

current_reg_stat_pos = pd.merge(current_reg_stat, Player_Position, how='left', left_on=['Player', 'Season'], right_on=['Player', 'Season'])
current_reg_stat_pos.isnull().sum()
current_reg_adv = pd.merge(current_reg_stat_pos, current_adv_stat, how='left', left_on=['Player', 'Season'], right_on=['Player', 'Season'])
# current_reg_stat_pos[current_reg_stat_pos['Pos'].isnull()].index.tolist()

# Clean up player string
reg_adv['Player'] = reg_adv['Player'].str.split(pat='\\').str[0]
current_reg_adv['Player'] = current_reg_adv['Player'].str.split(pat='\\').str[0]
# Cleaning up the dataset

reg_adv = reg_adv.drop(['Age_y', 'Tm_y', 'Lg_y', 'WS_y', 'G_y',
                        'GS_y', 'MP_y'], axis=1)
current_reg_adv = current_reg_adv.drop(['Age_y', 'Tm_y', 'Lg_y', 'WS_y', 'G_y',
                        'GS_y', 'MP_y'], axis=1)
replacing_col = {'Age_x':'Age',
                 'Tm_x':'Tm',
                 'Lg_x':'Lg,',
                 'WS_x':'WS',
                 'G_x':'G',
                 'GS_x':'GS',
                 'MP_x':'MP' }

reg_adv = reg_adv.rename(columns=replacing_col)
current_reg_adv = current_reg_adv.rename(columns=replacing_col)

# changing Player column formatting to match player_salary

period_player = reg_adv[reg_adv['Player'].str.contains('.', regex=False)]
period_removed_player = period_player['Player'].str.replace('.', '')
period_player_list = period_player['Player'].to_list()
period_removed_player_list = period_removed_player.to_list()

name_dict = {period_player_list[i]:period_removed_player_list[i] 
             for i in range(len(period_player_list))}

reg_adv.replace(to_replace={'Player': name_dict}, value=None, inplace=True)
current_reg_adv.replace(to_replace={'Player': name_dict}, value=None, inplace=True)
# Now adding salary and position

player_salary['Salary'] = player_salary['Salary'].replace('[\$,]', '', regex=True)
player_salary['Salary'] = player_salary['Salary'].str.strip()
player_salary['Salary'] = player_salary['Salary'].astype('float64')

player_salary.dtypes
reg_adv.dtypes

# merge into full data
full_data = pd.merge(left=reg_adv, right=player_salary, how='left',
                     left_on=['Player', 'Season'], right_on=['Player', 'Season'])
current_full_data = pd.merge(left=current_reg_adv, right=player_salary, how='left',
                     left_on=['Player', 'Season'], right_on=['Player', 'Season'])


# Now cleaning up data
full_data['3P%'] = full_data['3P%'].fillna(0)

full_data = full_data.drop(['Tm', 'Lg,'], axis=1)
current_full_data = current_full_data.drop(['Tm', 'Lg,'], axis=1)

full_data.isnull().sum()
current_full_data.isnull().sum()


# locate the 1 null vale and drop it
full_data[full_data['Salary'].isnull()].index.tolist()
current_full_data[current_full_data['Salary'].isnull()].index.tolist()

full_data.drop(608, inplace=True)
current_full_data.drop(134, inplace=True)

# EDA
describe = full_data.describe()
full_num = full_data.select_dtypes(include='number')
full_cat = full_data.select_dtypes(include='object')
full_num_current = current_full_data.select_dtypes(include='number')
full_cat_current = current_full_data.select_dtypes(include='object')
# The [:-1] is to remove correlation with itself
corr = full_num.corr()['Salary'][:-1]
corr.sort_values(ascending=False)

corr_current = full_num_current.corr()['Salary'][:-1]
corr_current.sort_values(ascending=False)

columns = ['VORP', 'PTS', 'FT', 'USG%', 'WS', 'TOV', 'PER', 'AST', 'TRB', 'TS%', 'DBPM']

# Check main variables
# high correlation between PTS,FT, and USG%. Drop FT and combine PTS and USG% to measure efficiency
# Keep VORP over WS
# Merge AST and TO to AST/TO
# age is a categorical data so we'll split into age groups and turn into dummies
# https://hoopshype.com/2018/12/31/nba-aging-curve-father-time-prime-lebron-james-decline/ using this, we turn Age into 5 buckets
# 19-22 development ,23-26 early prime, 27-31 prime, 32-35 Decline, 35+ geriatric

corrmat = full_data[columns].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True)

corrmat_current = current_full_data[columns].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True)

full_data['PPM'] = full_data['PTS'] / full_data['MP']
full_data['AST_TO'] = full_data['AST'] / full_data['TOV']

current_full_data['PPM'] = current_full_data['PTS'] / current_full_data['MP']
current_full_data['AST_TO'] = current_full_data['AST'] / current_full_data['TOV']
# full_data['Age'] = pd.cut(full_data['Age'], bins=[18, 22, 26, 31, 50], 
#                           labels=['Developing', 'Early Prime', 'Prime', 'Declining'])

columns_2 = ['VORP', 'PPM', 'AST_TO', 'PER', 'TRB', 'TS%','DBPM']
corrmat_2 = full_data[columns_2].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat_2, vmax=.8, square=True, annot=True)

corrmat_2_current = current_full_data[columns_2].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat_2_current, vmax=.8, square=True, annot=True)

# clean up pos column
full_data['Pos'] = full_data['Pos'].str.split(pat='-').str[0]


# Plotting our variables

# sns.distplot(full_data['VORP'])

# sns.catplot(x='Pos', y='Salary', data=full_data)

# Plot the most baic columns
sns.catplot(x='Pos', y='Salary', data=full_data)
sns.scatterplot(data=full_data, x='PTS', y='Salary')
sns.scatterplot(data=full_data, x='AST_TO', y='Salary', hue='Pos', style='Pos')

# plot all variables together
# sns.set()
# sns.pairplot(full_data[columns_2], size = 1)
# plt.show();

final_columns = ['Player', 'Salary','Season', 'VORP', 'PPM', 'AST_TO', 'TRB', 'TS%', 'DBPM']


# All columns that show skewness skew the same way as Salary. Normalize columns
# Try square root transformation ,log transform, recipricol transformation
# Come back and also standardize them once complete
# Log transforming causes NaN Data as can't deal with negative numbers
# Standardize then normalize?
    # Also Salary shows 2 peaks

normal_values = full_data[columns_2].values
min_max_scaler = preprocessing.MinMaxScaler()
normal_values_scaled = min_max_scaler.fit_transform(normal_values)
full_data[columns_2] = pd.DataFrame(normal_values_scaled)

normal_values_current = current_full_data[columns_2].values
min_max_scaler_current = preprocessing.MinMaxScaler()
normal_values_scaled_current = min_max_scaler.fit_transform(normal_values_current)
current_full_data[columns_2] = pd.DataFrame(normal_values_scaled_current)

# checking each variable for normal distribution
final_data = full_data[final_columns]
final_data.isnull().sum()

final_data_current = current_full_data[final_columns]
final_data_current.isnull().sum()
# drop row of NaN values
final_data[final_data['VORP'].isnull()].index.tolist()
final_data = final_data.drop(1087)

final_data_current[final_data_current['VORP'].isnull()].index.tolist()
final_data_current = final_data_current.drop(268)


# get dummies and set index as name in order to run analysis
final_data = final_data.set_index('Player')
final_data = pd.get_dummies(final_data, columns=['Season'])


final_data_current = pd.get_dummies(final_data_current, columns=['Season'])
final_data_current = final_data_current.set_index('Player')

def pct_cap(row):
    if row['Season_2017-18'] == 1:
        return row['Salary'] / 99093000
    if row['Season_2018-19'] == 1:
        return row['Salary'] / 101869000
    if row['Season_2019-20'] == 1:
        return row['Salary'] / 109140000
    if row['Season_2020-21'] == 1:
        return row['Salary'] / 109140000
    if row['Season_2021-22'] == 1:
        return row['Salary'] / 112414000
    return 1


final_data['Pct_Cap'] = final_data.apply(pct_cap, axis=1)
final_data_current['Pct_Cap'] = final_data_current['Salary'] / 112414000


conditions = [
        (final_data['Pct_Cap'] <= .02),
        (final_data['Pct_Cap'] > .02) & (final_data['Pct_Cap'] <= .04),
        (final_data['Pct_Cap'] > .04) & (final_data['Pct_Cap'] <= .07),
        (final_data['Pct_Cap'] > .07) & (final_data['Pct_Cap'] <= .10),
        (final_data['Pct_Cap'] > .10) & (final_data['Pct_Cap'] <= .13),
        (final_data['Pct_Cap'] > .13) & (final_data['Pct_Cap'] <= .16),
        (final_data['Pct_Cap'] > .16) & (final_data['Pct_Cap'] <= .19),
        (final_data['Pct_Cap'] > .19)
        ]

conditions_current = [
        (final_data_current['Pct_Cap'] <= .02),
        (final_data_current['Pct_Cap'] > .02) & (final_data_current['Pct_Cap'] <= .04),
        (final_data_current['Pct_Cap'] > .04) & (final_data_current['Pct_Cap'] <= .07),
        (final_data_current['Pct_Cap'] > .07) & (final_data_current['Pct_Cap'] <= .10),
        (final_data_current['Pct_Cap'] > .10) & (final_data_current['Pct_Cap'] <= .13),
        (final_data_current['Pct_Cap'] > .13) & (final_data_current['Pct_Cap'] <= .16),
        (final_data_current['Pct_Cap'] > .16) & (final_data_current['Pct_Cap'] <= .19),
        (final_data_current['Pct_Cap'] > .19)
        ]

values = ['1', '2', '3', '4', '5', '6', '7', '8']

final_data['Cluster'] = np.select(conditions, values)
final_data_current['Cluster'] = np.select(conditions_current, values)

# Since we normalized data, every column has 0 
# replace 0 with NaN to log transform
# for 3P columns, we need to seperate the 0 values since there are so many of them

# seperating 0s from 3P columns
# Creates a new column with the same length as the column, then sets all values to 0, then any row where final_data['3PA']>0 turns to 1
final_data = final_data.replace(0, np.nan)

final_data['Salary'] = np.log(final_data['Salary'])
final_data['VORP'] = np.log(final_data['VORP'])
final_data['PPM'] = np.log(final_data['PPM'])
final_data['AST_TO'] = np.log(final_data['AST_TO'])
final_data['TRB'] = np.log(final_data['TRB'])
final_data['DBPM'] = np.log(final_data['DBPM'])


final_data = final_data.replace(np.nan, 0)

final_data_current = final_data_current.replace(0, np.nan)

final_data_current['Salary'] = np.log(final_data_current['Salary'])
final_data_current['VORP'] = np.log(final_data_current['VORP'])
final_data_current['PPM'] = np.log(final_data_current['PPM'])
final_data_current['AST_TO'] = np.log(final_data_current['AST_TO'])
final_data_current['TRB'] = np.log(final_data_current['TRB'])
final_data_current['DBPM'] = np.log(final_data_current['DBPM'])


final_data_current = final_data_current.replace(np.nan, 0)

# sns.distplot(final_data['VORP'], fit=norm)
sns.distplot(final_data['PPM'], fit=norm)
# sns.distplot(final_data['AST_TO'], fit=norm)
# sns.distplot(final_data['TRB'], fit=norm)
# sns.distplot(final_data['DBPM'], fit=norm)
# sns.distplot(final_data_current['TS%'], fit=norm)
# sns.distplot(final_data['Salary'], fit=norm)

# Turn season dummy columns back to one so we can set it as index

dummy_cols = ['Season_2017-18','Season_2018-19', 'Season_2019-20', 'Season_2020-21']

def get_season(row):
    for c in final_data[dummy_cols].columns:
        if row[c]==1:
            return c


hope = final_data[dummy_cols].apply(get_season, axis=1)

final_data['place_hold'] = hope
final_data.columns
drop_col = ['Season_2017-18','Season_2018-19', 'Season_2019-20', 'Season_2020-21']

final_data = final_data.drop(drop_col, axis=1)

final_data[['drop', 'Season']] = final_data['place_hold'].str.split('_', 1, expand=True)
final_data = final_data.drop(['place_hold', 'drop'], axis=1)

# get rid of season column
final_data = final_data.set_index('Season', append=True)
final_data_current = final_data_current.drop('Season_2021-22', axis=1)


sns.distplot(x=final_data['Cluster'])

# Get training and test set

train_dataset = final_data.drop(['Salary','Pct_Cap','Cluster'], axis=1)
predict_dataset = final_data_current.drop(['Salary','Pct_Cap','Cluster'], axis=1)
Y = final_data['Cluster']

x_train, x_test, y_train, y_test = train_test_split(train_dataset, Y,  test_size=.2)

# y_train = y_train.squeeze()
# y_test = y_test.squeeze()
# Testing several different models

gnb = GaussianNB()
cv_gnb = cross_val_score(gnb, x_train, y_train, cv=10)
print(cv_gnb.mean())

lr = LogisticRegression(max_iter=2000)
cv_lr = cross_val_score(lr, x_train, y_train, cv=10)
print(cv_lr.mean())

dt = tree.DecisionTreeClassifier(random_state = 1)
cv_dt = cross_val_score(dt, x_train, y_train, cv=10)
print(cv_dt.mean())

knn = KNeighborsClassifier()
cv_knn = cross_val_score(knn, x_train, y_train, cv=10)
print(cv_knn.mean())

rf = RandomForestClassifier(random_state = 1)
cv_rf = cross_val_score(rf, x_train, y_train, cv=10)
print(cv_rf.mean())

svc = SVC(probability = True)
cv_svc = cross_val_score(svc, x_train, y_train, cv=10)
print(cv_svc.mean())

xgb = XGBClassifier(random_state =1)
cv_xgb = cross_val_score(xgb, x_train, y_train, cv=10)
print(cv_xgb.mean())

rf_fit = rf.fit(x_test,y_test)

rf_predict = rf.predict(predict_dataset)
output = pd.DataFrame({'Player':predict_dataset.index,
                        'Cluster':rf_predict.astype(int),
                        'Actual_Cluster':final_data_current['Cluster']},)
output.to_csv('Results.csv', index=False)


