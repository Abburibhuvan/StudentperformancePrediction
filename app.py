from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.ensemble as es
import sklearn.neural_network as nn
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

modelR = es.RandomForestClassifier(random_state=42).fit(X_train, y_train)
modelN = nn.MLPClassifier(activation="logistic", random_state=42).fit(X_train, y_train)

def acc(model): return round((model.predict(X_test) == y_test).mean() * 100, 1)

accuracies = {
    "Random Forest":      acc(modelR),
    "MLP Neural Network": acc(modelN),
}

label_map = {0: "High (H)", 1: "Medium (M)", 2: "Low (L)"}

@app.route("/")
def index():
    return render_template("index.html", accuracies=accuracies)

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()
    arr = np.array([
        int(body["raisedhands"]),
        int(body["visited_resources"]),
        int(body["discussion"]),
        int(body["absence"])
    ]).reshape(1, -1)

    predictions = {
        "Random Forest":      label_map.get(int(modelR.predict(arr)[0]), "Unknown"),
        "MLP Neural Network": label_map.get(int(modelN.predict(arr)[0]), "Unknown"),
    }
    return jsonify({
        "predictions": predictions,
        "final":       predictions["MLP Neural Network"],
        "accuracies":  accuracies
    })

if __name__ == "__main__":
    app.run(debug=True)
