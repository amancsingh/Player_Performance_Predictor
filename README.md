Player Performance Predictor and Optimal XI Recommender
1. Overview

Player Performance Predictor is a data-driven machine learning system designed to predict cricket player performance under specific match conditions and recommend an optimal playing XI using contextual analytics and constrained optimization.

The project addresses a major limitation of traditional sports analytics: reliance on historical averages and subjective judgment. Instead, it uses context-aware machine learning models trained on ball-by-ball data to generate probabilistic performance ranges and translate them into actionable selection decisions.

This project was developed as a final year B.Sc. Computer Science capstone project and demonstrates end-to-end implementation of data engineering, machine learning, backend development, and algorithmic optimization.

2. Problem Statement

Player selection in cricket is often influenced by the following limitations:

Career averages without context
Traditional metrics ignore venue, opposition, and match conditions.

Recency bias
Selection decisions overvalue the most recent match.

Subjective intuition
Human judgment introduces inconsistency and bias.

These approaches fail to adequately consider:

Venue-specific performance history

Opposition-specific matchups

Short-term form trends

Role balance within a team

The goal of this project is to replace subjective selection with a repeatable, explainable, and data-driven system.

3. Project Objectives

The core objectives of this project are:

Contextual prediction
Predict individual player performance using match-specific features.

Variance handling
Address the high volatility and class imbalance inherent in sports data.

Decision stability
Provide interpretable predictions suitable for real-world selection decisions.

Team optimization
Recommend a role-balanced optimal playing XI using quantitative impact scoring.

4. Core Features
4.1 Player Performance Prediction

This module provides context-aware performance forecasts with the following characteristics:

Output format
Performance is predicted as runs and wickets ranges (categorical bins).

Model separation
Independent models are used for:

Batting and Bowling

T20, ODI, and Test formats

Stability over precision
Classification-based predictions are preferred over fragile point estimates.

4.2 Contextual Feature Engineering

The predictive models incorporate the following feature groups:

Recent form metrics
Player performance over the last 10 matches.

Venue history
Ground-specific historical performance.

Opposition matchups
Player effectiveness against specific teams.

Career aggregates
Format-wise career statistics.

4.3 Optimal XI Recommendation System

The Optimal XI module converts individual predictions into team-level decisions:

Impact scoring
Each player is assigned a weighted impact score using:

Impact Score = Runs Score + (Wickets Score × 25)


Role-based constraints
Selection logic enforces:

Adequate batting depth

Sufficient bowling options

Inclusion of all-rounders

The output is a balanced, match-ready playing XI.

4.4 Web Application

The system is exposed through a web application with the following capabilities:

Interactive interface
Allows users to input match context and view predictions.

User authentication
Secure login and session management.

Prediction history
Persistent storage of past predictions.

API-driven inference
ML models are accessed via REST endpoints.

5. Machine Learning Approach
5.1 Model Design

Algorithm used
Random Forest Classifier.

Model count
Six independent models:

Batting and Bowling × T20 / ODI / Test.

5.2 Dataset

Data volume
Approximately 20,000 ball-by-ball match records.

Processing
Raw data transformed into structured, feature-rich datasets.

5.3 Prediction Strategy

Target type
Categorical performance bins.

Class imbalance handling
Managed using:

class_weight = balanced

5.4 Model Performance
Format	Batting Accuracy	Bowling Accuracy
Test	59.43%	73.19%
ODI	64.74%	66.39%
T20	60.70%	72.97%

Bowling models consistently outperform batting models due to lower variance and stronger dependence on contextual factors.

6. System Architecture
Backend

Language: Python

Framework: Flask (REST APIs)

ML Library: Scikit-learn

Database: SQLite

Model Storage: Joblib

Frontend

Markup: HTML5

Styling: Tailwind CSS

Scripting: JavaScript

Visualization: Chart.js

Data Processing

Libraries: Pandas, NumPy

Pipeline: Custom feature engineering workflows

7. Project Structure
├── app.py                 # Flask backend
├── models/                # Trained ML models (.joblib)
├── data/                  # SQLite database
├── notebooks/             # Model training and experiments
├── templates/             # HTML templates
├── static/                # CSS, JS, assets
├── requirements.txt
└── README.md

8. Running the Project Locally
Step 1: Clone the Repository
git clone https://github.com/your-username/player-performance-predictor.git
cd player-performance-predictor

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Start the Application
python app.py

Step 4: Access the Application
http://127.0.0.1:5000

9. Technical Highlights

End-to-end ML pipeline
From raw data ingestion to deployed inference.

Efficient imbalance handling
Avoids memory-intensive oversampling techniques.

Separation of concerns
Clear distinction between prediction and optimization layers.

Scalable architecture
Designed for API-based and cloud-ready systems.

Interpretability-first approach
Prioritizes stable decision-making over volatile precision.

10. Applicability Beyond Sports

The underlying architecture and methodology are transferable to:

FinTech risk scoring systems

Fraud and anomaly detection

User behavior modeling

Decision support systems

Context-aware recommendation engines

11. Future Enhancements

Live data ingestion via external APIs

Additional contextual features (pitch, weather, toss)

Ensemble modeling strategies

Cloud deployment (AWS / GCP)

Probabilistic output visualization

12. Author

Aman Chitranjan Singh
B.Sc. Computer Science
University of Mumbai

This project was developed as part of the final semester curriculum and reflects independent design, implementation, and evaluation.
