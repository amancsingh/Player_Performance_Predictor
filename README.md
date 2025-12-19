Player Performance Predictor and Optimal XI Recommender
Overview

Player Performance Predictor is a data-driven machine learning system designed to predict cricket player performance under specific match conditions and recommend an optimal playing XI using contextual analytics and constrained optimization.

The project addresses a key limitation of traditional sports analytics: reliance on historical averages and subjective judgment. Instead, it uses context-aware machine learning models trained on ball-by-ball data to generate probabilistic performance ranges and translate them into actionable selection decisions.

This project was developed as a final year B.Sc. Computer Science capstone project and demonstrates end-to-end implementation of data engineering, machine learning, backend development, and algorithmic optimization.

Problem Statement

Player selection in cricket is often influenced by:

Career averages without context

Recent performance bias

Human intuition and subjective judgment

These approaches fail to consider:

Venue-specific performance

Opposition-specific matchups

Short-term form trends

Role balance within a team

This project aims to replace subjective selection with a repeatable, explainable, and data-driven system.

Key Objectives

Predict individual player performance using contextual machine learning

Handle high variance and class imbalance in sports data

Provide stable, interpretable predictions suitable for decision-making

Recommend a role-balanced optimal XI based on predicted impact

Core Features
Player Performance Prediction

Predicts runs range and wickets range

Separate models for:

Batting and Bowling

T20, ODI, and Test formats

Predictions are returned as categorical bins instead of exact values to improve reliability

Contextual Feature Engineering

Models incorporate:

Player form over last 10 matches

Venue-specific historical performance

Opposition-specific performance

Career statistics (format-wise)

Optimal XI Recommendation

Converts predictions into a quantitative Impact Score

Weighted scoring formula:

Impact Score = Runs Score + (Wickets Score × 25)


Applies role-based constraints to ensure:

Adequate batting depth

Sufficient bowling options

Inclusion of all-rounders

Produces a balanced and match-ready playing XI

Web Application

Interactive web interface

User authentication

Prediction history tracking

End-to-end ML inference via REST APIs

Machine Learning Approach

Algorithm: Random Forest Classifier

Models:

6 independent models

Batting and Bowling × T20 / ODI / Test

Dataset:

Approximately 20,000 ball-by-ball match records

Target:

Performance bins (classification)

Imbalance Handling:

class_weight = balanced

Model Accuracy
Format	Batting Accuracy	Bowling Accuracy
Test	59.43%	73.19%
ODI	64.74%	66.39%
T20	60.70%	72.97%

Bowling models consistently outperform batting models due to lower variance and stronger dependency on contextual factors.

System Architecture

Backend:

Python

Flask (REST APIs)

Scikit-learn (ML inference)

SQLite (persistent storage)

Joblib (model serialization)

Frontend:

HTML5

Tailwind CSS

JavaScript

Chart.js for visualization

Data Processing:

Pandas

NumPy

Custom feature engineering pipelines

Project Structure
├── app.py                 # Flask backend
├── models/                # Trained ML models (.joblib)
├── data/                  # SQLite database
├── notebooks/             # Model training and experiments
├── templates/             # HTML templates
├── static/                # CSS, JS, assets
├── requirements.txt
└── README.md

How to Run Locally

Clone the repository:

git clone https://github.com/your-username/player-performance-predictor.git
cd player-performance-predictor


Install dependencies:

pip install -r requirements.txt


Start the application:

python app.py


Access in browser:

http://127.0.0.1:5000

Technical Highlights

Complete end-to-end machine learning pipeline from raw data to deployed inference

Practical handling of class imbalance without memory-intensive oversampling

Clear separation between prediction (machine learning) and decision-making (optimization)

Scalable backend architecture suitable for API-driven systems

Focus on interpretability and stability over fragile point predictions

Applicability Beyond Sports

Although implemented for cricket analytics, the system architecture and methodology are directly applicable to:

FinTech risk scoring systems

Fraud and anomaly detection

User behavior and churn prediction

Decision support systems

Context-aware recommendation engines

Future Scope

Live data ingestion via APIs

Additional contextual features such as pitch and weather

Ensemble modeling approaches

Cloud deployment (AWS / GCP)

Probability distribution visualization instead of single-bin output

Author

Aman Chitranjan Singh
B.Sc. Computer Science
University of Mumbai

This project was developed as part of the final semester curriculum and reflects independent design, implementation, and evaluation.
