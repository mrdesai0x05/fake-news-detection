# 📰 Fake News Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project implements a **Fake News Detection System** using **Machine Learning and Natural Language Processing (NLP)**.
It classifies news articles as **Fake 🛑 or Real ✅** based on textual content.

Additionally, it integrates **DevOps practices** like Git, GitHub, and **CI/CD pipelines** to ensure automation, reliability, and reproducibility.

---

## 🎯 Objectives

✔ Detect fake news using NLP techniques

✔ Compare multiple ML algorithms

✔ Automate workflows using CI/CD

✔ Maintain reproducible ML pipelines

✔ Demonstrate ML + DevOps integration

---

## ✨ Features

✔ Multiple ML Models (LR, DT, GBoost, RF)

✔ TF-IDF based feature extraction

✔ Real-time manual news testing

✔ High accuracy (up to 99%)

✔ Automated CI/CD pipeline

✔ Clean and modular code structure

---

## 🛠️ Tech Stack

### 🔹 Programming

* Python 3

### 🔹 Libraries

* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

### 🔹 Machine Learning

* Logistic Regression
* Decision Tree
* Gradient Boosting
* Random Forest

### 🔹 DevOps

* Git
* GitHub
* GitHub Actions (CI/CD)

---

## 📂 Dataset

Dataset consists of:

* `Fake.csv` → Fake news
* `True.csv` → Real news

Each contains:

* title
* text
* subject
* date

Labeling:

* `0 → Fake`
* `1 → Real`

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading

Load datasets using Pandas

### 2️⃣ Data Preprocessing

* Lowercasing
* Removing URLs, punctuation
* Cleaning text

### 3️⃣ Train-Test Split

Split dataset into training and testing

### 4️⃣ Feature Extraction

TF-IDF Vectorization

### 5️⃣ Model Training

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 98.31%   |
| Decision Tree       | 99.59%   |
| Gradient Boosting   | 99.56%   |
| Random Forest       | 98.66%   |

---

## 🧪 Manual Testing

```python
manual_testing("Breaking news text here")

# Sample Output:
LR Prediction : Fake News  
DT Prediction : Fake News  
GBC Prediction: Fake News  
RFC Prediction: Fake News  
```

---

## 🔁 CI/CD Integration

✔ Runs automatically on every GitHub push

✔ Installs dependencies

✔ Executes checks

✔ Ensures code reliability

---

## 🚀 How to Run

### 1️⃣ Install Python

Make sure Python 3 is installed

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3️⃣ Clone Repository

```bash
git clone https://github.com/your-repo-link/fake-news-detection.git
```

### 4️⃣ Run Project

```bash
python main.py
```

---

## 📊 Project Outcomes

✔ High accuracy fake news detection system

✔ Successfully implemented ML models

✔ Integrated CI/CD pipeline

✔ Built industry-relevant DevOps workflow

✔ Resume-ready project

---

## 🔮 Future Enhancements

🚀 Web app deployment (Streamlit / Flask)

🚀 Deep learning models (LSTM, BERT)

🚀 Docker integration

🚀 Cloud deployment (AWS / GCP)

🚀 Improved UI

---

## 👨‍🎓 Author

**Veeresh A**
Lovely Professional University

---

## 📚 Course

INT331

## 📅 Date

December 27, 2025

---

⭐ If you like this project, don't forget to star the repo!
