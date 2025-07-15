import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sys
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics
from sklearn.cluster import KMeans

# ======== Configuration ========
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize, precision=3)
sns.set(style="darkgrid")

# ======== Charger les données ========
train_path = "input/Train_data.csv"
test_path = "input/Test_data.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("Erreur : Fichiers de données introuvables.")
    exit()

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# ======== Nettoyage : supprimer colonnes inutiles ========
if 'num_outbound_cmds' in train.columns:
    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
if 'num_outbound_cmds' in test.columns:
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)

# ======== Normalisation des colonnes numériques ========
scaler = StandardScaler()
cols = train.select_dtypes(include=['float64', 'int64']).columns
sc_train = scaler.fit_transform(train[cols])
sc_test = scaler.transform(test[cols])
sc_traindf = pd.DataFrame(sc_train, columns=cols)
sc_testdf = pd.DataFrame(sc_test, columns=cols)

# ======== Encodage des colonnes catégoriques ========
encoder = LabelEncoder()
cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()
traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)

enctrain = traincat.drop(['class'], axis=1)
train_y = traincat['class']
train_x = pd.concat([sc_traindf, enctrain], axis=1)
test_df = pd.concat([sc_testdf, testcat], axis=1)

# ======== Sélection de caractéristiques ========
rfc = RandomForestClassifier()
rfc.fit(train_x, train_y)
rfe = RFE(rfc, n_features_to_select=15)
rfe = rfe.fit(train_x, train_y)
selected_features = train_x.columns[rfe.support_]
train_x = train_x[selected_features]
test_df = test_df[selected_features]

# ======== Séparation Train/Test ========
X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, train_size=0.70, random_state=2)

# ======== MÉTHODES SUPERVISÉES ========
supervised_models = [
    ('Régression Logistique', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, n_jobs=-1)),
    ('Naive Bayes (Bernoulli)', BernoulliNB()),
    ('K-Nearest Neighbors', KNeighborsClassifier(n_jobs=-1))
]

print("\n=========== MÉTHODES SUPERVISÉES ===========")
for name, model in supervised_models:
    model.fit(X_train, Y_train)
    pred_test = model.predict(X_test)
    acc = metrics.accuracy_score(Y_test, pred_test)
    print(f"\n{name} - Accuracy sur Test :", round(acc, 4))
    print("Matrice de confusion :\n", metrics.confusion_matrix(Y_test, pred_test))
    print("Classification Report :\n", metrics.classification_report(Y_test, pred_test))

# ======== MÉTHODE NON SUPERVISÉE : KMeans ========
print("\n=========== MÉTHODE NON SUPERVISÉE : KMeans ===========")
kmeans = KMeans(n_clusters=len(np.unique(Y_train)), random_state=0)
kmeans.fit(train_x)
labels = kmeans.predict(test_df)
print("Clusters prédits pour les 10 premières lignes :", labels[:10])

# ======== RÉGRESSION LINÉAIRE (Simulée) ========
# Crée une variable cible continue (exemple : somme des colonnes numériques)
train_reg = train_x.copy()
train_reg['target'] = train_x.sum(axis=1) + np.random.normal(0, 1, size=len(train_x))

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(train_x, train_reg['target'], test_size=0.3, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

mse = metrics.mean_squared_error(y_test_reg, y_pred_reg)
r2 = metrics.r2_score(y_test_reg, y_pred_reg)

print("\n=========== RÉGRESSION LINÉAIRE (Simulée) ===========")
print("Erreur quadratique moyenne (MSE) :", round(mse, 3))
print("Score R² :", round(r2, 3))

# ======== PREDICTIONS SUR TEST FINAL ========
print("\n=========== PRÉDICTIONS SUR DONNÉES TEST ===========")
for name, model in supervised_models:
    pred = model.predict(test_df)
    print(f"{name} - Prédictions (10 premières) :", pred[:10])
