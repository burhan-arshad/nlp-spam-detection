# SMS Spam Classifier

A machine learning application that classifies SMS messages as **Spam** or **Ham** using TF-IDF text representation and a Linear Support Vector Machine (LinearSVC).

## Live Demo

[Try the Live Demo](YOUR_STREAMLIT_APP_LINK)

## Overview

This project builds an end-to-end Natural Language Processing (NLP) pipeline for SMS spam detection.

The model learns patterns from labeled SMS messages and predicts whether a new message is legitimate or spam.

The final classifier combines:

* TF-IDF Vectorization
* Linear Support Vector Machine
* GridSearchCV for hyperparameter tuning

## Dataset

This project uses the SMS Spam Collection Dataset, containing labeled SMS messages classified as either ham or spam.

Dataset:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Machine Learning Pipeline

```text
SMS Message
     ↓
Train/Test Split
     ↓
TF-IDF Vectorization
     ↓
LinearSVC
     ↓
Prediction
     ↓
Spam / Ham
```

## Model Selection

Several classification algorithms were evaluated:

| Model               | Accuracy | Precision | Recall |
| ------------------- | -------: | --------: | -----: |
| Naive Bayes         |   96.41% |   100.00% | 73.15% |
| Random Forest       |   97.40% |   100.00% | 80.54% |
| Logistic Regression |   98.39% |    94.56% | 93.29% |
| Linear SVM          |   98.83% |    97.89% | 93.29% |

Linear SVM was selected for further optimization.

Because false positives are particularly important in spam detection, precision was used as the primary metric during hyperparameter tuning.

## Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to optimize the TF-IDF and LinearSVC parameters.

```python
param_grid = {
    'tfidf__max_features': [3000, 5000, 8000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10]
}
```

Best parameters:

```text
C = 10
max_features = 8000
ngram_range = (1, 1)
```

## Final Model Performance

| Metric         | Score |
| -------------- | ----: |
| Accuracy       |   99% |
| Spam Precision |   99% |
| Spam Recall    |   91% |
| Spam F1-Score  |   94% |
| Macro F1-Score |   97% |

## Why Precision Matters

In spam detection, a false positive occurs when a legitimate message is incorrectly classified as spam.

For example:

```text
Actual: Ham
Predicted: Spam
```

This can cause legitimate messages to be incorrectly filtered.

Therefore, precision was given particular importance during model tuning while still monitoring recall and F1-score.

## Web Application

The project includes a Streamlit frontend where users can:

* Enter an SMS message
* Classify it as Spam or Ham
* Test example messages
* View information about the underlying model

The trained TF-IDF vectorizer and LinearSVC classifier are stored together inside the serialized Pipeline:

```text
spam_classifier.pkl
```

This allows the application to process raw text directly without separately loading the TF-IDF vectorizer.

## Technologies Used

* Python
* Scikit-learn
* TF-IDF
* LinearSVC
* GridSearchCV
* Streamlit
* Joblib

## Project Structure

```text
sms-spam-classifier/
│
├── app.py
├── spam_classifier.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd sms-spam-classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

## Deployment

The application is deployed using Streamlit Community Cloud.

[Open the Live Application](YOUR_STREAMLIT_APP_LINK)

## Author

**Burhan Arshad**

Computer Science Student | Machine Learning & NLP

## License

This project is licensed under the MIT License.
