import numpy as np
import pandas as pd
import time
import seaborn as sns
from sklearn import preprocessing
from scipy.stats import f_oneway
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
import pickle
#########################################
#from regmilestone1 import *
testdata = pd.read_csv(r"E:\Machine Learning\ms1-games-tas-test-v1.csv")
y= testdata['Average User Rating']
ortestdata=testdata
#['User Rating Count', 'In-app Purchases', 'Age Rating', 'Size', 'Average purchases', 'Original year', 'Current year', 'lang count', 'FR', 'DE', 'JA', 'KO', 'ZH', 'ES', 'TH', 'RU', 'ID', 'PT', 'AR', 'Entertainment', 'Games', 'Productivity', 'Reference', 'Simulation', 'Casual', 'Puzzle', 'Action', 'Adventure', 'Board', 'Family', 'Kids & Family']


colnuames_test_data=list(testdata.columns.values)
#['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'User Rating Count', 'Price', 'In-app Purchases', 'Description', 'Developer', 'Age Rating', 'Languages', 'Size', 'Primary Genre', 'Genres', 'Original Release Date', 'Current Version Release Date', 'Average User Rating']
#drop
testdata.drop(columns=['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'Description','Price','Developer'], axis=1, inplace=True)

def avg_calc(text):
    if text != 0:
        lst = text.split(',')
        lst = [float(x) for x in lst]
        avg = sum(lst) / len(lst)
        return avg
    else:
        return 0
###
#Current Version Release Date,Original Release Date
testdata['Original year'] = testdata['Original Release Date'].str[-4:]

testdata['Current year'] = testdata['Current Version Release Date'].str[-4:]
testdata.drop(columns=['Original Release Date', 'Current Version Release Date'], axis=1, inplace=True)
#mean null cols
meanFile = open("means1.obj","rb")
meanDic=pickle.load(meanFile)
meanFile.close()
print(testdata)

def removesuffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text

testdata['Age Rating'] = testdata['Age Rating'].astype('str')
for i in range(len(testdata)):
    testdata.loc[i,'Age Rating']=removesuffix(testdata.loc[i, 'Age Rating'], "+")
testdata
testdata.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
null_cols = ['Age Rating','User Rating Count','Size','Current year','Original year']

for i in null_cols:
    testdata[i] = testdata[i].fillna(meanDic[i])
testdata

####
#In-app Purchases
testdata['In-app Purchases'].fillna(0, inplace=True)# sub in train get top accuracy
testdata['Average purchases'] = testdata['In-app Purchases'].apply(avg_calc)
testdata['In-app Purchases'] = testdata['In-app Purchases'].apply(lambda x: (len(list(pd.to_numeric(str(x).split(','))))))
#lang count
def new_feature(t):
    lst = t.split(',')
    lst = [str(x) for x in lst]
    return len(lst)

testdata['Languages'].fillna('EN', inplace=True)

testdata['lang count'] = testdata['Languages'].apply(new_feature)
testdata
#languages
data=testdata
uni = []
for i in range(data.shape[0]):
    column_index = data.columns.get_loc("Languages")
    lastindex = data.shape[1]

    s = data.iloc[i, column_index].split(',')
    for j in s:
        j = j.strip()
        if j not in uni:
            uni.append(j)

uni = [i.strip() for i in uni]

for i in range(len(uni)):
    data.insert(loc=i + lastindex, column=uni[i], value=0)

for i in range(data.shape[0]):
    s = data.iloc[i, column_index].split(',')
    s = [i.strip() for i in s]

    for j in range(len(uni)):
        if uni[j] in s:
            data.iloc[i, j + lastindex] = 1


data.drop(columns='Languages', axis=1, inplace=True)
####fill null values in geners and primary geners
modeFile = open("mode1.obj","rb")
modeDic=pickle.load(modeFile)
modeFile.close()
null_mode_cols = ['Genres','Primary Genre']

for i in null_mode_cols:
    data[i] = data[i].fillna(modeDic[i])
############################################
# Primary Genre
d = pd.get_dummies(data['Primary Genre'], dtype=int)
data = pd.concat([data, d], axis=1)
uni_primary = data['Primary Genre'].unique()

# Genre
uniGenre = []
data
for i in range(data.shape[0]):
    column_index = data.columns.get_loc("Genres")
    lastindex = data.shape[1]
    s = data.iloc[i, column_index].split(',')
    for j in s:
        j = j.strip()
        if j not in uniGenre:
            uniGenre.append(j)
print(uniGenre)
for i in uni_primary:
    if i not in uniGenre:
        print("Yes,", i, "is not in the primary genre column.")

have_no_col = []
for i in uniGenre:
    if i not in uni_primary:
        have_no_col.append(i)
print(have_no_col)
print(data.shape)
have_no_col = [i.strip() for i in have_no_col]
for i in range(len(have_no_col)):
    data.insert(loc=i + lastindex, column=have_no_col[i], value=0)

for i in range(data.shape[0]):
    s = data.iloc[i, column_index].split(',')
    s = [i.strip() for i in s]

    for j in range(len(have_no_col)):
        if have_no_col[j] in s:
            data.iloc[i, j + lastindex] = 1

data.drop(columns=['Primary Genre', 'Genres'], axis=1, inplace=True)


########
dropcolls=[]
colnames_testdata=list(data.columns.values)
teatdatashape=data.shape
colnames_corr_orginaldata=['User Rating Count', 'In-app Purchases', 'Size', 'Average purchases', 'Original year', 'Current year', 'lang count', 'FR', 'DE', 'JA', 'KO', 'ZH', 'ES', 'TH', 'RU', 'ID', 'PT', 'AR', 'Entertainment', 'Games', 'Productivity', 'Reference', 'Simulation', 'Casual', 'Puzzle', 'Action', 'Adventure', 'Board', 'Family', 'Kids & Family']
#drop cols that does not exist in orginaldata after feature selection
for a in colnames_testdata:
   if not (a  in colnames_corr_orginaldata):
     dropcolls.append(a)
dropcolls
###drop from test data
data.drop(columns=dropcolls, axis=1, inplace=True)
#added cols ==> categorical clos in (orginaldata after feature selection) and doesnot exist in (test data)
addedcols=[]
for a in colnames_corr_orginaldata:

   if not (a  in colnames_testdata):
     addedcols.append(a)

addedcols
#add added cols  to test data fill with 0's
for a in addedcols:
 data[a] = 0
#################
colnames_testdata=list(data.columns.values)
teatdatashape=data.shape
###########################
###scale
#make cols in test data same order as orginal data to applay  saved feature scale
data = data.reindex(colnames_corr_orginaldata , axis=1)
scaler = pickle.load(open("scalerReg.obj","rb"))
data=scaler.transform(data)
data=pd.DataFrame(data)
Xtest=data
####################
#1 st model GradientBoosting_model
# Start_time =time.time()
# pickled_model = pickle.load(open('GradientBoosting_model.pkl', 'rb'))
# predictions = pickled_model.predict(Xtest)
# end_time =time.time()
# total=end_time-Start_time
# print('Mean Square Error for GradientBoosting model: ', metrics.mean_squared_error(y, predictions))#
# print('Accuracy of ' + 'GradientBoosting' + ': {:.2f}%'.format(r2_score(y, predictions) * 100))
# print(len(predictions))
# print('time of ' + 'GradientBoosting' , total)
#############2nd model' XGBRegressor_model'

Start_time =time.time()
pickled_model = pickle.load(open('XGBRegressor_model.pkl', 'rb'))
predictions = pickled_model.predict(Xtest)
end_time =time.time()
total2=end_time-Start_time
print('Mean Square Error for XGBRegressor_model: ', metrics.mean_squared_error(y, predictions))#
print('Accuracy of ' + 'XGBRegressor_model' + ': {:.2f}%'.format(r2_score(y, predictions) * 100))
print('time of ' + 'XGBRegressor_model' , total2)





