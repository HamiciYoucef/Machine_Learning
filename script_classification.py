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
from sklearn.metrics import adjusted_rand_score, completeness_score, confusion_matrix

# ======== Configuration ========
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# ======== Charger les données ========
train_path = "input/Train_data.csv"
test_path = "input/Test_data.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("Erreur : Fichiers de données introuvables.")
    exit()

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# ======== Nettoyage : supprimer colonnes inutiles ========
for df in [train, test]:
    if 'num_outbound_cmds' in df.columns:
        df.drop(['num_outbound_cmds'], axis=1, inplace=True)

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

# ======== Apprentissage Supervisé ========
print("\n====== Apprentissage Supervisé ======")
metrics_names = ['accuracy', 'precision', 'recall', 'f1-score']
scores_summary = {metric: [] for metric in metrics_names}
model_labels = []
supervised_models = [
    ('Logistic Regression', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)),
    ('Naive Bayes', BernoulliNB()),
    ('KNN', KNeighborsClassifier())
]

trained_models = []

for name, model in supervised_models:
    model.fit(X_train, Y_train)
    trained_models.append((name, model))
    pred = model.predict(X_test)
    report = metrics.classification_report(Y_test, pred, output_dict=True)

    acc = metrics.accuracy_score(Y_test, pred)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    scores_summary['accuracy'].append(acc)
    scores_summary['precision'].append(precision)
    scores_summary['recall'].append(recall)
    scores_summary['f1-score'].append(f1)
    model_labels.append(name)

    print(f"\n{name} - Scores détaillés :")
    print(f"Accuracy  : {acc:.4f}\nPrecision : {precision:.4f}\nRecall    : {recall:.4f}\nF1-Score  : {f1:.4f}")
    print(metrics.classification_report(Y_test, pred))

# ======== Affichage détaillé par modèle ========
for idx, name in enumerate(model_labels):
    values = [scores_summary[m][idx] for m in metrics_names]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(metrics_names, values, color=['skyblue', 'lightgreen', 'orange', 'violet'])
    plt.ylim(0, 1.05)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center')
    plt.title(f"Scores pour {name}")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

# ======== Matrices de confusion ========
for name, model in trained_models:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matrice de confusion - {name}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.show()

# ======== Comparaison globale des métriques ========
for metric in metrics_names:
    plt.figure(figsize=(7, 4))
    values = scores_summary[metric]
    bars = plt.bar(model_labels, values, color='cornflowerblue')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center')
    plt.ylim(0, 1.05)
    plt.title(f"Comparaison globale - {metric.capitalize()}")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

# ======== Apprentissage Non Supervisé (KMeans) ========
print("\n====== Apprentissage Non Supervisé ======")
kmeans = KMeans(n_clusters=len(np.unique(Y_train)), random_state=0)
kmeans.fit(X_train)
ari = adjusted_rand_score(Y_train, kmeans.labels_)
completeness = completeness_score(Y_train, kmeans.labels_)
print(f"KMeans ARI (Adjusted Rand Index): {ari:.4f}")
print(f"KMeans Completeness Score      : {completeness:.4f}")

# ======== Régression Linéaire ========
print("\n====== Régression ======")
reg_model = LinearRegression()
reg_model.fit(X_train, Y_train)
Y_pred_reg = reg_model.predict(X_test)
mse = metrics.mean_squared_error(Y_test, Y_pred_reg)
r2 = metrics.r2_score(Y_test, Y_pred_reg)
print(f"MSE : {mse:.4f}\nR²  : {r2:.4f}")

# ======== Graphe Comparatif KMeans + Regression ========
plt.figure(figsize=(8, 5))
plt.bar(['KMeans (ARI)', 'Régression (R²)'], [ari, r2], color=['salmon', 'teal'])
plt.title("KMeans (non supervisé) vs Régression linéaire")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
