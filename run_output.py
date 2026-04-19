import pandas as pd
import seaborn as sb
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import warnings as w
w.filterwarnings('ignore')

data = pd.read_csv("AI-Data.csv")

# Save correlation heatmap
plt.figure(figsize=(12, 8))
sb.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output_heatmap.png")
plt.close()
print("Correlation heatmap saved as output_heatmap.png")

# Save all graphs
graphs = [
    ("Marks Class Count Graph", lambda: sb.countplot(x='Class', data=data, order=['L', 'M', 'H'])),
    ("Marks Class Semester-wise", lambda: sb.countplot(x='Semester', hue='Class', data=data, hue_order=['L','M','H'])),
    ("Marks Class Gender-wise", lambda: sb.countplot(x='gender', hue='Class', data=data, order=['M','F'], hue_order=['L','M','H'])),
    ("Marks Class Nationality-wise", lambda: sb.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L','M','H'])),
    ("Marks Class Grade-wise", lambda: sb.countplot(x='GradeID', hue='Class', data=data, order=['G-02','G-04','G-05','G-06','G-07','G-08','G-09','G-10','G-11','G-12'], hue_order=['L','M','H'])),
    ("Marks Class Section-wise", lambda: sb.countplot(x='SectionID', hue='Class', data=data, hue_order=['L','M','H'])),
    ("Marks Class Topic-wise", lambda: sb.countplot(x='Topic', hue='Class', data=data, hue_order=['L','M','H'])),
    ("Marks Class Stage-wise", lambda: sb.countplot(x='StageID', hue='Class', data=data, hue_order=['L','M','H'])),
    ("Marks Class Absent Days-wise", lambda: sb.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L','M','H'])),
]

for i, (title, plot_fn) in enumerate(graphs, 1):
    plt.figure(figsize=(10, 6))
    plot_fn()
    plt.title(title)
    plt.tight_layout()
    fname = f"output_graph_{i}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Graph {i} saved: {fname}")

# Drop columns
drop_cols = ["gender","StageID","GradeID","NationalITy","PlaceofBirth","SectionID",
             "Topic","Semester","Relation","ParentschoolSatisfaction","ParentAnsweringSurvey","AnnouncementsView"]
data = data.drop(columns=drop_cols)

data = u.shuffle(data, random_state=42)

for column in data.columns:
    if data[column].dtype not in ['int64', 'float64', 'int32', 'float32']:
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))

ind = int(len(data) * 0.70)
feats = data.values[:, 0:4]
lbls = data.values[:, 4]
feats_Train = feats[0:ind]
feats_Test = feats[(ind+1):len(feats)]
lbls_Train = lbls[0:ind]
lbls_Test = lbls[(ind+1):len(lbls)]

print("\n" + "="*50)
print("ML MODEL ACCURACY RESULTS")
print("="*50)

# Decision Tree
modelD = tr.DecisionTreeClassifier(random_state=42)
modelD.fit(feats_Train, lbls_Train)
lbls_predD = modelD.predict(feats_Test)
countD = sum(a == b for a, b in zip(lbls_Test, lbls_predD))
print("\nDecision Tree Classifier:")
print(m.classification_report(lbls_Test, lbls_predD))
print("Accuracy:", round(countD / len(lbls_Test), 3))

# Random Forest
modelR = es.RandomForestClassifier(random_state=42)
modelR.fit(feats_Train, lbls_Train)
lbls_predR = modelR.predict(feats_Test)
countR = sum(a == b for a, b in zip(lbls_Test, lbls_predR))
print("\nRandom Forest Classifier:")
print(m.classification_report(lbls_Test, lbls_predR))
print("Accuracy:", round(countR / len(lbls_Test), 3))

# Perceptron
modelP = lm.Perceptron(random_state=42)
modelP.fit(feats_Train, lbls_Train)
lbls_predP = modelP.predict(feats_Test)
countP = sum(a == b for a, b in zip(lbls_Test, lbls_predP))
print("\nLinear Model Perceptron:")
print(m.classification_report(lbls_Test, lbls_predP))
print("Accuracy:", round(countP / len(lbls_Test), 3))

# Logistic Regression
modelL = lm.LogisticRegression(random_state=42)
modelL.fit(feats_Train, lbls_Train)
lbls_predL = modelL.predict(feats_Test)
countL = sum(a == b for a, b in zip(lbls_Test, lbls_predL))
print("\nLogistic Regression:")
print(m.classification_report(lbls_Test, lbls_predL))
print("Accuracy:", round(countL / len(lbls_Test), 3))

# MLP Neural Network
modelN = nn.MLPClassifier(activation="logistic", random_state=42)
modelN.fit(feats_Train, lbls_Train)
lbls_predN = modelN.predict(feats_Test)
countN = sum(a == b for a, b in zip(lbls_Test, lbls_predN))
print("\nMLP Neural Network Classifier:")
print(m.classification_report(lbls_Test, lbls_predN))
print("Accuracy:", round(countN / len(lbls_Test), 3))

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Decision Tree      : {round(countD/len(lbls_Test), 3)}")
print(f"Random Forest      : {round(countR/len(lbls_Test), 3)}")
print(f"Perceptron         : {round(countP/len(lbls_Test), 3)}")
print(f"Logistic Regression: {round(countL/len(lbls_Test), 3)}")
print(f"MLP Neural Network : {round(countN/len(lbls_Test), 3)}")
