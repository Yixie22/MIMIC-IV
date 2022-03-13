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


# data---------------------------------------
df = pd.read_csv('/home/e0205142/project/MIMIC_assignment1/data/train_corrected_encoded_imputed_oversample.csv')
df = df.drop(columns = ['Glascow coma scale eye opening', 'Glascow coma scale motor response',  'Glascow coma scale total'])

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(X.shape)
print(y.shape)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# selected features---------------------------------
sorted_idx = np.array([ 0,  1,  6,  5,  9, 10,  4, 11,  3,  8,  2,  7])
sorted_idx = sorted_idx[::-1]

# models-------------------------------------------------
models = [
          ('LR', LogisticRegression()), 
          ('DT', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('AdaBoost', AdaBoostClassifier(n_estimators=1000)),
          ('GBM', GradientBoostingClassifier(n_estimators=1000)),
          ('SVM', SVC(probability=True))
        ]

LR_params = [{'C': [ 10, 1.0, 0.1, 0.01]}]
DT_params = [{}]
RF_params = [{'n_estimators': [100, 500]}]
Gradient_params = [{'learning_rate': [0.001, 0.01, 0.1], 'n_estimators': [500, 1000], 'subsample': [0.5, 0.7, 1.0], 'max_depth': [3, 7, 9]}]
Ada_params = [{'learning_rate': [0.001, 0.01, 0.1], 'n_estimators': [500, 1000], 'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=5)]}]
SVM_params =[{'kernel': ['linear', 'poly', 'rbf'], 'C': [100, 10, 1.0, 0.1, 0.001]}]

my_grid ={'LR': LR_params,
         'DT': DT_params,
         'RF': RF_params,
         'AdaBoost': Ada_params,
         'GBM': Gradient_params,
         'SVM': SVM_params}

# evaluate all the models
train_acc, test_acc, train_auc, test_auc = [], [], [], []
names = []
cv_results = []
best_params = []
clfs = []
runtimes = []

for name, model in models:
    start_time = time.time()
    
    X_train_selected = X_train[:, sorted_idx[:4]]
    X_test_selected = X_test[:, sorted_idx[:4]]
    
    clf = GridSearchCV(model, my_grid[name], cv = 5, scoring='roc_auc', n_jobs=10)
    clf.fit(X_train_selected, y_train)

    y_train_pred = clf.predict(X_train_selected)
    y_train_score = clf.predict_proba(X_train_selected)
    y_test_pred = clf.predict(X_test_selected)
    y_test_score = clf.predict_proba(X_test_selected)
    #cv_results.append(cv_result)
    
    train_acc.append(accuracy_score(y_train, y_train_pred))
    test_acc.append(accuracy_score(y_test, y_test_pred))
    train_auc.append(roc_auc_score(y_train, y_train_score[:, 1]))
    test_auc.append(roc_auc_score(y_test, y_test_score[:, 1]))
    
    best_params.append(clf.best_params_)
    clfs.append(clf)
    
    names.append(name)
    
    runtime = (time.time() - start_time)
    runtimes.append(runtime)
    print("---%s finished in %s seconds ---" % (name, runtime))
    print('>%s: test AUC %.3f' % (name, roc_auc_score(y_test, y_test_score[:, 1])))

scores_df = pd.DataFrame({'train_acc':train_acc, 
                          'test_acc':test_acc, 
                          'train_auc':train_auc,
                          'test_auc':test_auc, 
                          'runtime(s)': runtimes}, index=names)
print(scores_df)
scores_df.to_csv('/home/e0205142/project/MIMIC_assignment1/results/final_models_evaluation.csv', index=True)

with open('/home/e0205142/project/MIMIC_assignment1/results/final_models_best_params.txt', 'w') as f:
    for name, para in zip(names, best_params):
        print(name, para, file=f)
        
with open('/home/e0205142/project/MIMIC_assignment1/results/final_clfs.pkl','wb') as f :
    pickle.dump(clfs, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()