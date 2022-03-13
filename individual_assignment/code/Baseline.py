import pandas as pd
import numpy as np
import time, pickle

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate, KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('/home/e0205142/project/MIMIC_assignment1/data/train_corrected_encoded_imputed.csv')

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

print(X.shape)
print(y.shape)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# # define oversampling strategy
# oversample = RandomOverSampler(sampling_strategy='minority')
# # fit and apply the transform
# X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)

# Models---------------------------------------
models = [
          ('LR', LogisticRegression(class_weight = 'balanced')), 
          ('DT', DecisionTreeClassifier(class_weight = 'balanced')),
          ('RF', RandomForestClassifier(class_weight = 'balanced')),
          ('AdaBoost', AdaBoostClassifier(n_estimators=1000)),
          ('GBM', GradientBoostingClassifier(n_estimators=1000)),
          ('SVM', SVC(probability=True, class_weight = 'balanced'))
        ]

# Parameter tuning grids-------------------------
LR_params = [{'C': [ 10, 1.0, 0.1, 0.01]}]
DT_params = [{}]
RF_params = [{'n_estimators': [50, 100, 500]}]
Gradient_params = [{'learning_rate': [0.001, 0.01, 0.1], 'n_estimators': [500, 1000], 'subsample': [0.5, 0.7, 1.0], 'max_depth': [3, 7, 9]}]
Ada_params = [{'learning_rate': [0.001, 0.01, 0.1], 'n_estimators': [500, 1000], 'base_estimator': [DecisionTreeClassifier(max_depth=1, class_weight = 'balanced'), DecisionTreeClassifier(max_depth=5, class_weight = 'balanced')]}]
SVM_params =[{'kernel': ['linear', 'poly', 'rbf'], 'C': [100, 10, 1.0, 0.1, 0.001]}]

my_grid ={'LR': LR_params,
         'DT': DT_params,
         'RF': RF_params,
         'AdaBoost': Ada_params,
         'GBM': Gradient_params,
         'SVM': SVM_params}

# Model Training and Evaluation-------------------
train_acc, test_acc, train_auc, test_auc = [], [], [], []
names = []
runtimes = []
best_params = []
clfs = []

for name, model in models:
    start_time = time.time()
    
    clf = GridSearchCV(model, my_grid[name], cv = 5, scoring='roc_auc', n_jobs=10)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_train_score = clf.predict_proba(X_train)
    y_test_pred = clf.predict(X_test)
    y_test_score = clf.predict_proba(X_test)
    #cv_results.append(cv_result)
    runtime = (time.time() - start_time)
    
    train_acc.append(accuracy_score(y_train, y_train_pred))
    test_acc.append(accuracy_score(y_test, y_test_pred))
    train_auc.append(roc_auc_score(y_train, y_train_score[:, 1]))
    test_auc.append(roc_auc_score(y_test, y_test_score[:, 1]))
    
    best_params.append(clf.best_params_)
    clfs.append(clf)
    
    names.append(name)
    runtimes.append(runtime)
    print("---%s finished in %s seconds ---" % (name, runtime))
    print('>%s: test AUC %.3f' % (name, roc_auc_score(y_test, y_test_score[:, 1])))

scores_df = pd.DataFrame({'train_acc':train_acc, 
                          'test_acc':test_acc, 
                          'train_auc':train_auc,
                          'test_auc':test_auc, 
                          'runtime(s)':runtimes}, index=names)
# save data
# 1. scores and running time
scores_df.to_csv('/home/e0205142/project/MIMIC_assignment1/results/baseline_after_all_preprocess_before_feature_selection.csv', index=True)

# 2. model parameters
with open('/home/e0205142/project/MIMIC_assignment1/results/all_models_best_params.txt', 'w') as f:
    for name, para in zip(names, best_params):
        print(name, para, file=f)

# 3. models
with open('/home/e0205142/project/MIMIC_assignment1/results/clfs.pkl','wb') as f :
    pickle.dump(clfs, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
