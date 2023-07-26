from discription import *
import numpy as np
import pandas as pd
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

# read dataset
data = key_word_data

# understand the data
print(data.columns)
print(data.head(5))
print(data.info())
print(data.describe())
print(data.isnull().sum())
print(data.shape)

# preprocessing
# URL ,ID ,NAME ,Subtitle ,ICON URL
print("The Number of Unique URLs = ", data['URL'].nunique())
print("The Number of Unique IDs = ", data['ID'].nunique())
print("The Number of Unique Names = ", data['Name'].nunique())
data.drop_duplicates(subset=["URL", "ID", "Name"], keep="first", inplace=True)
print("The shape of data after removing duplicates is", data.shape)
data.drop(columns=['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'Description'], axis=1, inplace=True)
# print(data.head())

# In-app purchase
data['In-app Purchases'].fillna(0, inplace=True)


def avg_calc(text):
    if text != 0:
        lst = text.split(',')
        lst = [float(x) for x in lst]
        avg = sum(lst) / len(lst)
        return avg
    else:
        return 0


data['Average purchases'] = data['In-app Purchases'].apply(avg_calc)
data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x: (len(list(pd.to_numeric(str(x).split(','))))))

# Developer
print("The Number of Unique Developers = ", data['Developer'].nunique())
data['Developer'] = data['Developer'].apply(lambda x: (len(list((str(x).split(','))))))

# Age rating
data['Age Rating'] = pd.factorize(data['Age Rating'])[0] + 1
print(data['Age Rating'].head(3))

# Original Date
data['Original year'] = data['Original Release Date'].str[-4:]
data['Original year'] = pd.to_numeric(data['Original year'])
print(data['Original year'].head(3))

# Current Date
data['Current year'] = data['Current Version Release Date'].str[-4:]
data['Current year'] = pd.to_numeric(data['Current year'])
print(data['Current year'].head(3))

# get new feature from Original and Current year
data['diff years'] = np.abs(data['Current year'] - data['Original year'])
print(data['diff years'].head(3))
data.drop(columns=['Original Release Date', 'Current Version Release Date'], axis=1, inplace=True)

# languages
missvalue = data['Languages'].mode()
data['Languages'] = data['Languages'].fillna(missvalue[0])
print("the shape ", data.shape)


def new_feature(t):
    lst = t.split(',')
    lst = [str(x) for x in lst]
    return len(lst)


data['lang count'] = data['Languages'].apply(new_feature)
print(data['lang count'].head(3))

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
print(data.columns)

# Primary Genre
d = pd.get_dummies(data['Primary Genre'], dtype=int)
data = pd.concat([data, d], axis=1)
uni_primary = data['Primary Genre'].unique()

# Genre
uniGenre = []
print(data.columns)
print(data.shape)
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
# Description


#########################
# anova test ==> categorical cols
for i in uni_primary:
    uni.append(i)
for c in uniGenre:
    uni.append(c)
print(uni)

columns = ['Age Rating']
columns = columns + uni

data = data.loc[:, (data == 1).mean() < 1]
# data = data.loc[:, (data==0).mean() <= .5]

colnames = list(data.columns.values)

anova_drop = []
for c in columns:
    if (c in colnames):
        col = pd.DataFrame(data[c])
        col.insert(1, 'Average User Rating', data.loc[:, 'Average User Rating'], True)
        CategoryGroupLists = col.groupby(c)['Average User Rating'].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)
        AnovaResults = AnovaResults[1]
        if AnovaResults > 0.05:
            anova_drop.append(c)

print("The columns that has no relationship:", anova_drop)
data.drop(columns=anova_drop, axis=1, inplace=True)

###############
# correlation
# print(data.corr())
plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True)
plt.show()
corr = data.corr(method='pearson')
most = abs(corr['Average User Rating'])

less_feature = corr.index[abs(corr['Average User Rating']) < 0.03]
less_feature = less_feature.values.tolist()
for a in columns:
    if a in less_feature:
        less_feature.remove(a)
print(less_feature)
data.drop(columns=less_feature, axis=1, inplace=True)
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True)
plt.show()
print(data)
X = data.drop(columns='Average User Rating', axis=1, inplace=False)
y = pd.DataFrame(data['Average User Rating'])
print(pd.DataFrame(X).columns)
# Feature scale
# Min-Max Normalization
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

# Data after preprocessing
X = pd.DataFrame(X)
y = pd.DataFrame(y)
####################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=30)

models = {
    "XGB Regressor": XGBRegressor(),
    "Support Vector Regressor": SVR(),
    "Gradient Boosting Regression": GradientBoostingRegressor(),
    "Linear Regression": linear_model.LinearRegression()
}

model_rf = []
for name, model in models.items():
    model_rf.append(model.fit(X_train, y_train))
    print(name + " trained.")

counter = 0
pred = []
predtrain = []
for name, model in models.items():
    pred.append(model_rf[counter].predict(X_test))
    predtrain.append(model_rf[counter].predict(X_train))

    print()
    print('Accuracy of ' + name + ': {:.2f}%'.format(r2_score(y_test, pred[counter]) * 100))

    print('Mean Square Error of ' + name, metrics.mean_squared_error(y_test, pred[counter]))
    counter += 1

counter = 0
for name, model in models.items():
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, pred[counter])
    mae = mean_absolute_error(y_test, pred[counter])
    r2 = r2_score(y_test, pred[counter])

    # Set up plot parameters
    metric_names = ['MSE', 'MAE', 'R-squared']
    metric_values = [mse, mae, r2]
    colors = ['b', 'g', 'r']
    linestyles = ['-', '--', ':']
    labels = ['Model: ' + metric_names[i] for i in range(len(metric_names))]
    x_label = 'Metric'
    y_label = 'Value'
    title = 'Model Evaluation Metrics of ' + name + " model"
    # Generate plots
    for i in range(len(metric_names)):
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot([0, 0], [1, 0], 'k--', color='gray', alpha=0.5)
        plt.plot([1, 1], [0, 1], 'k--', color='gray', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.plot(metric_values[i], metric_values[i], 'o', color=colors[i])
        plt.annotate(labels[i] + ' = {:.2f}'.format(metric_values[i]), xy=(metric_values[i], metric_values[i]),
                     xytext=(metric_values[i] + 0.05, metric_values[i] - 0.05))
        if i == 2:  # add regression line plot for R-squared metric
            plt.figure(figsize=(15, 13))
            x = np.array(list(range(len(X_test))))  # Add this line to generate x-axis values
            plt.scatter(x, y_test, color='blue', label='Data')
            plt.plot(x, pred[counter], color='black', linewidth=0.75, label='Predictions')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.title('Regression line plot for ' + name + ' model')
            plt.legend(loc='best')
            plt.show()
        else:
            plt.legend(loc="lower right")
            plt.show()
    counter += 1
