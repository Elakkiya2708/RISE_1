from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# create folder automatically
os.makedirs("static/images", exist_ok=True)

app = Flask(__name__)

# Load dataset
df = pd.read_csv("data/customer_data.csv", on_bad_lines='skip')

# Preprocessing
df.dropna(inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Train simple churn model
X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

@app.route('/')
def login():
    return render_template("login.html")

@app.route('/dashboard', methods=['POST'])
def dashboard():
    
    # 📊 Visualization 1 - Churn Count
    plt.figure()
    sns.countplot(x="Churn", data=df)
    plt.title("Customer Churn Distribution")
    plt.savefig("static/images/churn_plot.png")
    plt.close()

    # 📊 Visualization 2 - Correlation
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation")
    plt.savefig("static/images/correlation.png")
    plt.close()

    # 📊 Visualization 3 - Segmentation
    plt.figure()
    sns.scatterplot(x=df.iloc[:,0], y=df.iloc[:,1], hue=y)
    plt.title("Customer Segmentation")
    plt.savefig("static/images/segmentation.png")
    plt.close()

    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)