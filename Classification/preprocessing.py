import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from scipy.stats import chi2_contingency


def pre(data, Y):
    # preprocessing

    # Rate column "Y"
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    # URL ,ID ,NAME ,Subtitle ,ICON URL
    print("The Number of Unique URLs = ", data['URL'].nunique())
    print("The Number of Unique IDs = ", data['ID'].nunique())
    print("The Number of Unique Names = ", data['Name'].nunique())
    print("The Number of Unique Subtitle = ", data['Subtitle'].nunique())
    print("The Number of Unique Icon url = ", data['Icon URL'].nunique())
    data.drop(columns=['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'Description'], axis=1, inplace=True)

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
    # print(data.columns)

    # Primary Genre
    d = pd.get_dummies(data['Primary Genre'], dtype=int)
    data = pd.concat([data, d], axis=1)
    uni_primary = data['Primary Genre'].unique()
    print(data.columns)
    # Genre
    uniGenre = []
    # print(data.columns)
    # print(data.shape)
    data['Genres'] = data['Genres'].fillna("Games")
    for i in range(data.shape[0]):
        column_index = data.columns.get_loc("Genres")
        lastindex = data.shape[1]
        print(data.iloc[i, column_index])
        s = data.iloc[i, column_index].split(',')
        for j in s:
            j = j.strip()
            if j not in uniGenre:
                uniGenre.append(j)
    # print(uniGenre)
    for i in uni_primary:
        if i not in uniGenre:
            print("Yes,", i, "is not in the primary genre column.")

    have_no_col = []
    for i in uniGenre:
        if i not in uni_primary:
            have_no_col.append(i)
    # print(have_no_col)
    # print(data.shape)
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

    # feature-selection
    ##############
    for i in uni_primary:
        uni.append(i)
    for c in uniGenre:
        uni.append(c)
    # print(uni)

    columns = ['Age Rating']
    columns = columns + uni

    return columns, data, Y


def feature_selection(columns, data, Y):
    data = data.loc[:, (data == 1).mean() < 1]

    catcols = columns
    numericcols = list(data.columns.values)
    for a in catcols:

        if a in numericcols:
            numericcols.remove(a)
    ############################

    data = data.loc[:, (data == 1).mean() < 1]
    # data = data.loc[:, (data==0).mean() <= .5]

    colnames = list(data.columns.values)
    # anova for x=numeric  y=categorical
    anova_drop = []
    for c in numericcols:
        # cc=c
        if (c in colnames):
            col = pd.DataFrame(data[c])
            col.insert(1, 'Rate', Y, True)
            CategoryGroupLists = col.groupby(c)['Rate'].apply(list)
            AnovaResults = f_oneway(*CategoryGroupLists)
            AnovaResults = AnovaResults[1]
            if AnovaResults > 0.01:
                anova_drop.append(c)
    data.drop(columns=anova_drop, axis=1, inplace=True)

    #######################################

    # Cross tabulation between categorical x and categorical y
    chi2_drop = []
    colnames = list(data.columns.values)
    for c in catcols:
        # cc=c
        ca = c
        if (c in colnames):
            CrosstabResult = pd.crosstab(index=data[c], columns=Y)
            print(CrosstabResult)
            # Performing Chi-sq test
            ChiSqResult = chi2_contingency(CrosstabResult)
            ChiSqResult = ChiSqResult[1]

            if ChiSqResult > 0.001:
                chi2_drop.append(c)
    data.drop(columns=chi2_drop, axis=1, inplace=True)
    return data, Y


def feature_scale(data):
    # Min-Max Normalization
    scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(data)
    return data
