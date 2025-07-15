import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree, metrics
import itertools
import os

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize, precision=3)
sns.set(style="darkgrid")

# Load data
train_path = "input/Train_data.csv"
test_path = "input/Test_data.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("Erreur : Fichiers de données introuvables.")
    exit()

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Drop redundant column
if 'num_outbound_cmds' in train.columns:
    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
if 'num_outbound_cmds' in test.columns:
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)

# Scale numerical attributes
scaler = StandardScaler()
cols = train.select_dtypes(include=['float64','int64']).columns
sc_train = scaler.fit_transform(train[cols])
sc_test = scaler.transform(test[cols])
sc_traindf = pd.DataFrame(sc_train, columns=cols)
sc_testdf = pd.DataFrame(sc_test, columns=cols)

# Encode categorical attributes
encoder = LabelEncoder()
cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()
traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)
enctrain = traincat.drop(['class'], axis=1)
cat_Ytrain = traincat[['class']].copy()

train_x = pd.concat([sc_traindf, enctrain], axis=1)
train_y = train['class']
test_df = pd.concat([sc_testdf, testcat], axis=1)

# Feature selection
rfc = RandomForestClassifier()
rfc.fit(train_x, train_y)
rfe = RFE(rfc, n_features_to_select=15)
rfe = rfe.fit(train_x, train_y)
feature_map = [(i, v) for i, v in zip(rfe.get_support(), train_x.columns)]
selected_features = [v for i, v in feature_map if i]
train_x = train_x[selected_features]
test_df = test_df[selected_features]

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, train_size=0.70, random_state=2)

# Train models
KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(X_train, Y_train)

LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(X_train, Y_train)

BNB_Classifier = BernoulliNB()
BNB_Classifier.fit(X_train, Y_train)

DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(X_train, Y_train)

# Evaluate models
models = [
    ('Naive Baye Classifier', BNB_Classifier),
    ('Decision Tree Classifier', DTC_Classifier),
    ('KNeighborsClassifier', KNN_Classifier),
    ('LogisticRegression', LGR_Classifier)
]

for name, model in models:
    scores = cross_val_score(model, X_train, Y_train, cv=10)
    accuracy = metrics.accuracy_score(Y_train, model.predict(X_train))
    print(f"\n==== {name} - Training Evaluation ====")
    print("Cross Validation Mean Score:", scores.mean())
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", metrics.confusion_matrix(Y_train, model.predict(X_train)))
    print("Classification Report:\n", metrics.classification_report(Y_train, model.predict(X_train)))

# Validation on test set
for name, model in models:
    accuracy = metrics.accuracy_score(Y_test, model.predict(X_test))
    print(f"\n==== {name} - Test Evaluation ====")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", metrics.confusion_matrix(Y_test, model.predict(X_test)))
    print("Classification Report:\n", metrics.classification_report(Y_test, model.predict(X_test)))

# Prediction on real test data
pred_knn = KNN_Classifier.predict(test_df)
pred_NB = BNB_Classifier.predict(test_df)
pred_log = LGR_Classifier.predict(test_df)
pred_dt = DTC_Classifier.predict(test_df)

print("\nExemple de prédictions (KNN) :", pred_knn[:10])
