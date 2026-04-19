from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import sklearn.metrics as m
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# --- Train models on startup ---
data = pd.read_csv("AI-Data.csv")
drop_cols = ["gender","StageID","GradeID","NationalITy","PlaceofBirth","SectionID",
             "Topic","Semester","Relation","ParentschoolSatisfaction","ParentAnsweringSurvey","AnnouncementsView"]
data = data.drop(columns=drop_cols)
data = u.shuffle(data, random_state=42)

for col in data.columns:
    if data[col].dtype not in ['int64','float64','int32','float32']:
        le = pp.LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

ind = int(len(data) * 0.70)
feats = data.values[:, 0:4]
lbls  = data.values[:, 4]
X_train, X_test = feats[:ind], feats[ind+1:]
y_train, y_test = lbls[:ind],  lbls[ind+1:]

modelD = tr.DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
modelR = es.RandomForestClassifier(random_state=42).fit(X_train, y_train)
modelP = lm.Perceptron(random_state=42).fit(X_train, y_train)
modelL = lm.LogisticRegression(random_state=42).fit(X_train, y_train)
modelN = nn.MLPClassifier(activation="logistic", random_state=42).fit(X_train, y_train)

def acc(model): return round((model.predict(X_test) == y_test).mean() * 100, 1)

accuracies = {
    "Decision Tree":       acc(modelD),
    "Random Forest":       acc(modelR),
    "Perceptron":          acc(modelP),
    "Logistic Regression": acc(modelL),
    "MLP Neural Network":  acc(modelN),
}

label_map = {0: "High (H)", 1: "Medium (M)", 2: "Low (L)"}

@app.route("/output_graph_<int:n>.png")
def graph(n):
    return send_from_directory(".", f"output_graph_{n}.png")

@app.route("/output_heatmap.png")
def heatmap():
    return send_from_directory(".", "output_heatmap.png")

@app.route("/")
def index():
    return render_template("index.html", accuracies=accuracies)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    arr = np.array([
        int(data["raisedhands"]),
        int(data["visited_resources"]),
        int(data["discussion"]),
        int(data["absence"])
    ]).reshape(1, -1)

    predictions = {
        "Decision Tree":       label_map.get(int(modelD.predict(arr)[0]), "Unknown"),
        "Random Forest":       label_map.get(int(modelR.predict(arr)[0]), "Unknown"),
        "Perceptron":          label_map.get(int(modelP.predict(arr)[0]), "Unknown"),
        "Logistic Regression": label_map.get(int(modelL.predict(arr)[0]), "Unknown"),
        "MLP Neural Network":  label_map.get(int(modelN.predict(arr)[0]), "Unknown"),
    }
    return jsonify({
        "predictions": predictions,
        "final": predictions["MLP Neural Network"],
        "accuracies": accuracies
    })

if __name__ == "__main__":
    app.run(debug=True)
