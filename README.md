📰 Fake News Detection System - ML with CI/CD Integration

📌 Project Overview

This repository contains our project for the CSE316 course at Lovely Professional University, submitted as part of Academic Task-1. Our team developed a Fake News Detection System that analyzes online news content and classifies it as Fake or Real using Machine Learning and Natural Language Processing (NLP).

The project also integrates DevOps practices such as Git, GitHub, and CI/CD pipelines, ensuring automation, reproducibility, and efficient model validation.

🔹 What It Does

Our Fake News Detection System allows users to:

Select different machine learning models for prediction.
Input news text and classify it as Fake or Real.
Train and evaluate multiple ML algorithms.
Automatically validate workflows using CI/CD pipelines.
Calculate and display model performance metrics:

✅ Accuracy

✅ Precision

✅ Recall

✅ F1-score

✨** Features**

✔ Multiple ML Algorithms: Logistic Regression, Decision Tree, Gradient Boosting, Random Forest
✔ Text Processing Pipeline: Cleans and preprocesses real-world news data
✔ Feature Extraction: Uses TF-IDF vectorization
✔ Model Comparison: Evaluates performance across different classifiers
✔ CI/CD Integration: Automates testing and validation using GitHub Actions
✔ Manual Testing: Allows users to input custom news text for prediction


🚀 How to Run It

Follow these steps to set up and run the project on your system:

1️⃣ Install Python 3

Download and install Python 3 from python.org
Ensure Python is added to your system PATH

2️⃣ Install Required Libraries

Open terminal or command prompt and run:
pip install pandas numpy scikit-learn matplotlib seaborn

3️⃣ Clone or Download the Repository

git clone https://github.com/your-repo-link/fake-news-detection.git
Or download ZIP from GitHub and extract it


4️⃣ Navigate to Project Directory

cd fake-news-detection

5️⃣ Run the Project

python main.py


📂 Project Structure

The project is divided into multiple modules for better maintainability:

📌 Main Script (main.py): Integrates all components and runs the system
📊 Data Processing Module: Handles data cleaning and preprocessing
🤖 ML Model Module: Implements and trains different classifiers
📈 Evaluation Module: Computes performance metrics and results

👥 Team Members and Contributions

Our team consists of members responsible for different parts of the system:
Veeresh A (12319252): Developed ML models and CI/CD integration


🛠️ Technologies Used

Python 3: Core programming language

Libraries Used:

Pandas, NumPy → Data handling

Scikit-learn → Machine Learning models

Matplotlib, Seaborn → Visualization

ML Techniques:

TF-IDF Vectorization

Supervised Learning Algorithms

DevOps Tools:

Git → Version control

GitHub → Collaboration

GitHub Actions → CI/CD automation

🔄 Development Process

Repository Setup: Created GitHub repository for collaboration

Module Development:
Each component (preprocessing, models, evaluation) was developed separately

Integration & Testing:
Ensured all modules work together seamlessly

CI/CD Setup:
Automated testing and validation using GitHub Actions

Documentation:
Created this README.md for project explanation

🧪 Notes

The system supports real-time prediction using manual input
CI/CD pipeline runs automatically on every push
Dataset includes labeled fake and real news articles
Preprocessing removes noise such as URLs, punctuation, and symbols

🔮 Future Improvements

✅ Deploy as a web application (Streamlit / Flask)
✅ Add deep learning models (LSTM, BERT)
✅ Integrate Docker for containerization
✅ Enable cloud deployment (AWS / GCP)
✅ Improve UI for better user interaction

📌 GitHub Contributions

We made multiple commits in this repository to track development progress.
Check commit history for detailed implementation steps!

📚 Course: INT331
🏫 Institution: Lovely Professional University
📅 Date: December 27, 2025
