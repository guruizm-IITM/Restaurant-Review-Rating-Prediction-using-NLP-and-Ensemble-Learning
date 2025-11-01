# Sentiment Classification using TF-IDF and Logistic Regression

## 📘 Project Overview

This project focuses on **emotion detection from text data** using a
**TF-IDF Vectorizer** and **Logistic Regression** model. The goal is to
accurately classify emotional expressions into predefined categories
such as *joy, sadness, anger, fear, surprise, and love*. The model is
trained on preprocessed text, vectorized using TF-IDF, and fine-tuned
for multi-class classification.

------------------------------------------------------------------------

## 🎯 Objectives

-   Preprocess and clean raw text data for emotion classification.
-   Apply TF-IDF vectorization to transform text into numerical
    features.
-   Train and evaluate a Logistic Regression model for multi-class
    classification.
-   Ensure predictions align with original emotion labels.
-   Build a reproducible, clean pipeline suitable for deployment or
    further experimentation.

------------------------------------------------------------------------

## 🧠 Model Workflow

1.  **Data Cleaning** -- Handle missing values, remove extra spaces, and
    sanitize column names.\
2.  **Text Preprocessing** -- Tokenization, lowercasing, and optional
    stopword removal.\
3.  **Feature Extraction** -- Convert text into numerical features using
    TF-IDF.\
4.  **Model Training** -- Train a Logistic Regression classifier on the
    transformed dataset.\
5.  **Prediction & Evaluation** -- Predict emotions on test data and map
    them back to original labels.

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   **Python 3.9+**
-   **scikit-learn**
-   **pandas**
-   **numpy**
-   **nltk** (for preprocessing, if needed)

------------------------------------------------------------------------

## 📊 Evaluation Metrics

-   **Accuracy**
-   **Precision**
-   **Recall**
-   **F1-Score**
-   **Confusion Matrix** for detailed analysis

------------------------------------------------------------------------

## 🏗️ Repository Structure

    emotion-detection-tfidf-logreg/
    │
    ├── data/
    │   ├── train.csv
    │   ├── test.csv
    │
    ├── notebooks/
    │   ├── eda_and_preprocessing.ipynb
    │   ├── model_training.ipynb
    │
    ├── src/
    │   ├── preprocess.py
    │   ├── train_model.py
    │   ├── predict.py
    │
    ├── models/
    │   ├── tfidf_vectorizer.pkl
    │   ├── logistic_regression_model.pkl
    │
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

## 🚀 How to Run

### 1️⃣ Clone the Repository

``` bash
git clone https://github.com/<your-username>/emotion-detection-tfidf-logreg.git
cd emotion-detection-tfidf-logreg
```

### 2️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

### 3️⃣ Run the Pipeline

``` bash
python src/train_model.py
```

### 4️⃣ Predict Emotions

``` bash
python src/predict.py --input "I am feeling great today!"
```

------------------------------------------------------------------------

## 📈 Results

The model demonstrates strong accuracy and generalization capability
across diverse emotion classes. TF-IDF with Logistic Regression provides
a solid baseline for emotion detection tasks.

------------------------------------------------------------------------

## 💡 Future Improvements

-   Experiment with **transformer-based embeddings** (e.g., BERT).\
-   Add **cross-validation** and **hyperparameter tuning**.\
-   Extend to **multi-label emotion detection**.\
-   Deploy via **Flask** or **FastAPI** for real-time inference.

------------------------------------------------------------------------

## 👨‍💻 Author

**Abhishek Guru**\
Data Science & AI, IIT Madras\
Passionate about NLP, Kaggle, and applied machine learning.

------------------------------------------------------------------------

## 🏷️ License

This project is licensed under the **MIT License**.
