# fake-news-detection
A real-time fake news detection system that analyzes online content to identify and flag misleading or false information using machine learning and natural language processing.
📰 Fake News Detection Using Machine Learning with CI/CD Integration
📌 Project Overview

This project implements a Fake News Detection System using Machine Learning and integrates it into a DevOps workflow using Git, GitHub, and CI/CD pipelines.
The system classifies news articles as Fake or Real based on textual content using multiple ML models.

The goal of this project is not only to build an accurate ML model but also to demonstrate version control, automation, reproducibility, and continuous integration, aligning with modern DevOps practices.

🎯 Objectives

Detect fake news using NLP and Machine Learning

Compare multiple ML classifiers

Automate model validation using CI/CD

Maintain reproducible ML workflows using Git

Demonstrate DevOps concepts in an ML project

🛠️ Tools & Technologies

Programming Language: Python

Libraries:

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

ML Models Used:

Logistic Regression

Decision Tree

Gradient Boosting

Random forest

Vectorization: TF-IDF

DevOps Tools:

Git

GitHub

GitHub Actions (CI/CD)

📂 Dataset

The dataset consists of two CSV files:

Fake.csv → Fake news articles

True.csv → Real news articles

Each dataset contains:

title

text

subject

date

A new column class is added:

0 → Fake News

1 → Real News

⚙️ Project Workflow
1️⃣ Data Loading
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

2️⃣ Data Preprocessing

Lowercasing text

Removing URLs, punctuation, numbers

Removing HTML tags

Cleaning unnecessary symbols

Custom preprocessing function:

def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\W', ' ', text)
    return text

3️⃣ Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25
)

4️⃣ Feature Extraction (TF-IDF)
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

5️⃣ Model Training & Evaluation
Model	Accuracy
Logistic Regression	98.31%
Decision Tree	99.59%
Gradient Boosting	99.56%
Random Forest	98.66%

Decision Tree and Gradient Boosting performed the best.

6️⃣ Classification Report Example
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1-score: 1.00

🧪 Manual Testing

Users can input any news text and receive predictions from all models.

manual_testing("Breaking news text here")

Sample Output:
LR Prediction : Fake News
DT Prediction : Fake News
GBC Prediction: Fake News
RFC Prediction: Fake News

🔁 CI/CD Integration

Every push to GitHub triggers GitHub Actions

Automatically:

Sets up Python environment

Installs dependencies

Runs sanity checks

Ensures reliability and reproducibility

📌 Project Outcomes

Successfully built a high-accuracy fake news classifier

Integrated Machine Learning with DevOps workflows

Implemented CI/CD automation

Gained hands-on experience with real-world ML + DevOps integration

Developed a resume-worthy academic project

🚀 Future Enhancements

Deploy as a web application (Streamlit / Flask)

Add deep learning models (LSTM / BERT)

Integrate Docker for containerization

Cloud deployment (AWS / GCP)

👨‍🎓 Author

Veeresh A

Lovely Professional University
