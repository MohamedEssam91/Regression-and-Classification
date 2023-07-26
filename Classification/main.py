from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocessing import *
import time
import matplotlib.ticker as ticker
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
#from discription import *

# read dataset

data = pd.read_csv(
     r"E:\Machine Learning\milestone2\milestone2\games-classification-dataset.csv")
#data=keyworddata
# URL ,ID ,NAME ,Subtitle ,ICON URL
data.drop_duplicates(subset=["URL", "ID", "Name", "Subtitle", "Icon URL"], keep="first", inplace=True)

# print(data.head())
X = data.drop(columns='Rate', axis=1, inplace=False)
y = pd.DataFrame(data['Rate'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32, shuffle=True)
cols, X_train, y_train = pre(X_train, y_train)
X_train, y_train = feature_selection(cols, X_train, y_train)
selected_columns = pd.DataFrame(X_train).columns
X_train = feature_scale(X_train)
cols, X_test, y_test = pre(X_test, y_test)
print(selected_columns)
print(X_test.columns)

##########
# the code here
# select the feature according to selected_columns
colummns = pd.DataFrame(X_test).columns
X_test = pd.DataFrame(X_test)
for i in selected_columns:
    if i not in colummns:
        X_test[i] = 0

X_test = X_test[selected_columns]
print(selected_columns)
print(X_test.columns)
##########

# Uncomment this below line after selecting the features from the X
X_test = feature_scale(X_test)

# create a list of models to evaluate
models = [
    ('Decision Tree',
     DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                            max_features=None, min_impurity_decrease=0.0, random_state=42)),
    # ('Decision Tree',
    #  DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=2, min_samples_leaf=1,
    #                         max_features=None, min_impurity_decrease=0.0, random_state=42)),
    # ('Decision Tree',
    #  DecisionTreeClassifier(criterion='entropy', max_depth=100, min_samples_split=2, min_samples_leaf=1,
    #                         max_features=None, min_impurity_decrease=0.0, random_state=42)),
    # ('Decision Tree',
    #  DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    #                         max_features='log2', min_impurity_decrease=0.0, random_state=42)),
    # ('Decision Tree',
    #  DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    #                         max_features='sqrt', min_impurity_decrease=0.0, random_state=42)),
    # ('Decision Tree',
    #  DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    #                         max_features='auto', min_impurity_decrease=0.0, random_state=42)),

    ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=4,
                                             max_features='sqrt', random_state=32)),
    # ('Random Forest', RandomForestClassifier(n_estimators=2, max_depth=10, min_samples_split=5, min_samples_leaf=4,
    #                                          max_features='sqrt', random_state=32)),
    # ('Random Forest', RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=5, min_samples_leaf=4,
    #                                          max_features='sqrt', random_state=32)),
    # ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_split=5, min_samples_leaf=4,
    #                                          max_features='sqrt', random_state=32)),    
    # ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=19, min_samples_split=5, min_samples_leaf=4,
    #                                                                                       max_features='sqrt', random_state=32)),    
    # ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth = 6, min_samples_split=5, min_samples_leaf=4,
    #                                                                                                                                    max_features='sqrt', random_state=32)),
    ("naive", GaussianNB()),
    # ("Neural network", MLPClassifier(hidden_layer_sizes=(10, 5), learning_rate = "adaptive", batch_size = 32)),
    # ("Neural network", MLPClassifier(hidden_layer_sizes=(100, 10), learning_rate = "adaptive", batch_size = 32)),
    # ("Neural network", MLPClassifier(hidden_layer_sizes=(50, 10), learning_rate = "adaptive", batch_size = 32)),
    ("Neural network", MLPClassifier(hidden_layer_sizes=(100, 10), learning_rate = "constant", batch_size = 32)),
    # ("Neural network", MLPClassifier(hidden_layer_sizes=(100, 10), learning_rate = "invscaling", batch_size = 32)),
    ('Gradient Boosting', GradientBoostingClassifier(loss='deviance', n_estimators=150, subsample=.3,
                                                    warm_start=False, validation_fraction=0.5,
                                                    max_depth=6, learning_rate=0.01,
                                                    min_samples_split=7))
    # ('Gradient Boosting', GradientBoostingClassifier(loss='deviance', n_estimators=150, subsample=.3,
    #                                                  warm_start=False, validation_fraction=0.5,
    #                                                  max_depth=6, learning_rate=0.0001,
    #                                                  min_samples_split=7))
    # ('Gradient Boosting', GradientBoostingClassifier(loss='deviance', n_estimators=150, subsample=.3,
    #                                                  warm_start=False, validation_fraction=0.5,
    #                                                  max_depth=6, learning_rate=0.1,
    #                                                  min_samples_split=7))
    # ('Gradient Boosting', GradientBoostingClassifier(loss='deviance', n_estimators=3, subsample=.3,
    #                                                  warm_start=False, validation_fraction=0.5,
    #                                                  max_depth=6, learning_rate=0.01,
    #                                                  min_samples_split=7))
    # ('Gradient Boosting', GradientBoostingClassifier(loss='deviance', n_estimators=10, subsample=.3,
    #                                                  warm_start=False, validation_fraction=0.5,
    #                                                  max_depth=6, learning_rate=0.01,
    #                                                  min_samples_split=7))
    # ('Gradient Boosting', GradientBoostingClassifier(loss='deviance', n_estimators=20, subsample=.3,
    #                                                  warm_start=False, validation_fraction=0.5,
    #                                                  max_depth=6, learning_rate=0.01,
    #                                                  min_samples_split=7))
]
# parameters = {
#     "n_estimators":[50,100,200],
#     "max_depth":[3,5,6,7],
#     "learning_rate":[0.01,0.001,0.0001]
# }

# clf =GradientBoostingClassifier()
# cv = RandomizedSearchCV(clf,parameters,n_iter = 30 , cv=5, random_state=42)
# cv.fit(X_train,y_train)

# print ("Best Parameters: {}".format(cv.best_params_))
# y_pred = cv.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy of gradient boosting with Randomized search: {:2f}".format(accuracy))



# Train the model and measure the training time
model_rf = []
total_time_of_each_model_for_training = []
for name, model in models:
    start_time = time.time()
    model_rf.append(model.fit(X_train, y_train))
    print(name + " trained.")
    end_time = time.time()
    training_time = end_time - start_time
    print("The total training time that the", name, "takes = ", training_time)
    print()
    total_time_of_each_model_for_training.append(training_time)
# print("the model rf :", model_rf)

# Test the model and measure the testing time
counter = 0
pred = []
predtrain = []
y_predict_proba = []
total_time_of_each_model_for_testing = []
print("The testing Model Details")
for name, model in models:
    start_time = time.time()
    pred.append(model_rf[counter].predict(X_test))
    end_time = time.time()
    testing_time = end_time - start_time
    print("The total testing time that the", name, "takes = ", testing_time)
    total_time_of_each_model_for_testing.append(testing_time)
    predtrain.append(model_rf[counter].predict(X_train))
    print('Accuracy of ' + name + ': {:.2f}%'.format(accuracy_score(y_test, pred[counter]) * 100))
    print('Mean Square Error of ' + name, metrics.mean_squared_error(y_test, pred[counter]))
    print()
    y_predict_proba.append(model_rf[counter].predict_proba(X_test))
    counter += 1

# Saving the results in PKL Files for future use
import pickle

for name, model in models:
    pickle.dump(model, open(name + ".pkl", "wb"))

# Showing the Classification Report and Confusion Matrix
counter = 0
for name, model in models:
    print(name + " Model Metrics:")
    print(classification_report(y_test, pred[counter]))
    print(confusion_matrix(y_test, pred[counter]))
    print()
    counter += 1

model = [
    "Decision Tree",
    "Random Forest",
    "Gaussian Naive",
    "Neural network",
    "Gradient Boosting"]

# Visualizing the ROC And GINI for y_test and prediction of the x_test


lb = LabelBinarizer()
lb.fit(y_test)

auc_score_test = []
gini = []
for y_pred in pred:
    y_pred_binarized = lb.transform(y_pred)
    auc_score = roc_auc_score(lb.transform(y_test), y_pred_binarized, average=None)
    auc_score_test.append(auc_score)
    gini.append(2 * auc_score - 1)
    # da = pd.DataFrame(y_pred)
    # print("The y-pred", da.nunique())

counter = 0
for y_pred in pred:
    y_pred_binarized = lb.transform(y_pred)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(lb.classes_.shape[0]):
        fpr[i], tpr[i], _ = roc_curve(lb.transform(y_test)[:, i], y_pred_binarized[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    colors = ['blue', 'orange', 'green']
    for i, color in zip(range(lb.classes_.shape[0]), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='%s Class AUC = %0.4f, GINI = %0.4f' % (i, roc_auc[i], gini[counter][i]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(model[counter])
    plt.show()
    counter += 1

models = [
    "Decision Tree",
    "Random Forest",
    "Gaussian Naive",
    "Neural network",
    "Gradient Boosting"]
# Classification Accuracy Bar Graph
accuracy = [accuracy_score(y_test, pred[0]), accuracy_score(y_test, pred[1]), accuracy_score(y_test, pred[2]),
            accuracy_score(y_test, pred[3]),accuracy_score(y_test, pred[4])]

plt.bar(models, accuracy, color='b')
plt.title('Classification Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
# Add text labels for each bar
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.xticks(rotation=45)
plt.show()

# Total Training Time Bar Graph
training_time = [total_time_of_each_model_for_training[0], total_time_of_each_model_for_training[1],
                 total_time_of_each_model_for_training[2], total_time_of_each_model_for_training[3], total_time_of_each_model_for_training[4]]
plt.bar(models, training_time, color='r')
plt.title('Total Training Time')
plt.xlabel('Model')
plt.ylabel('Time (s)')
# Set y-axis ticks and format as floats
yticks = [i * 0.8 for i in range(0, 21)]
plt.yticks(yticks)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
# Add text labels for each bar
for i, v in enumerate(training_time):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.xticks(rotation=45)
plt.show()

# Total Test Time Bar Graph
testing_time = [total_time_of_each_model_for_testing[0], total_time_of_each_model_for_testing[1],
                total_time_of_each_model_for_testing[2], total_time_of_each_model_for_testing[3], total_time_of_each_model_for_testing[4]]
plt.bar(models, testing_time, color='g')
plt.title('Total Test Time')
plt.xlabel('Model')
plt.ylabel('Time (s)')
# Set y-axis ticks and format as floats
yticks = [i * 0.005 for i in range(0, 10)]
plt.yticks(yticks)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# Add text labels for each bar
for i, v in enumerate(testing_time):
    plt.text(i, v + 0.001, f"{v:.3f}", ha='center')
plt.xticks(rotation=45)
plt.show()
