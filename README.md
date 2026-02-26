#  Spam Detection System using Deep Learning (RNN)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

End-to-end deep learning project to classify emails as Spam or Ham using NLP and a Bidirectional LSTM model.

---

##  Project Overview

This project implements an end-to-end Spam Detection System using Natural Language Processing (NLP) and Deep Learning.

The system classifies text messages into:

- **Ham (Legitimate message)**
- **Spam (Suspicious message)**

It includes complete preprocessing, model training, testing, and deployment using **Streamlit** for real-time predictions.

---

## 🎯 Problem Statement

Spam messages are a major issue in email and messaging platforms.
The objective of this project is to:

- Clean and preprocess raw text messages
- Convert text into numerical vectors
- Train a deep learning model
- Deploy a real-time spam classification web app

---

##  Dataset

- Dataset: SMS Spam Collection (CSV format)
- Encoding: latin_1
- Labels:
  - ham → 1
  - spam → 0

The dataset contains text messages labeled as spam or legitimate.

---

##  Model Architecture

The model is built using:

- Embedding Layer
- Recurrent Neural Network (RNN)
- Dense Output Layer

### Key Settings:

- Vocabulary Size: 5500
- Padding Length: 953
- Activation: Sigmoid
- Output: Binary Classification

---

##  NLP & Processing Pipeline

### 1️. Text Cleaning

- Convert to lowercase
- Remove punctuation
- Remove stopwords
- Apply lemmatization

### 2️. Text Vectorization

- One-hot encoding
- Sequence padding
- Fixed-length numerical representation

### 3️. Model Prediction

- Output probability between 0 and 1
- If probability > 0.5 → HAM
- If probability ≤ 0.5 → SPAM

---

##  Deployment (Streamlit App)

The model is deployed using **Streamlit**.

### Web App Features:

- User inputs a message
- Automatic preprocessing
- Real-time prediction
- Confidence score display
- Progress bar visualization
- View processed text option

---

## Project Structure
```
Spam_Analysis/
│
├── app/
│   └── app.py                # Streamlit web application
│
├── data/
│   └── spam.csv              # Dataset (SMS Spam Collection)
│
├── models/
│   └── spam_model.pkl        # Trained RNN model
│
├── src/
│   ├── train.py              # Model training script
│   ├── preprocess.py         # Text preprocessing pipeline
│   └── logger.py             # Logging system
│
├── notebooks/
│   └── spam_detection.ipynb  # EDA and model experimentation
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- NLTK
- Scikit-learn
- Streamlit

---

##  How to Run the Project

### 1️. Clone Repository
```bash
git clone https://github.com/your-username/Spam_Analysis.git
cd Spam_Analysis
```

### 2️. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️. Run Streamlit App
```bash
streamlit run app/app.py
```

---

##  Sample Input
```
Congratulations! You have won a free iPhone. Click the link now!
```

### Output:
```
SPAM (Suspicious)
Confidence: 97.82%
```

---

## Applications

- Email spam filtering
- SMS spam detection
- Fraud message detection
- Customer communication filtering
- Enterprise email security

---

##  Key Highlights

✔ End-to-end Deep Learning project

✔ Custom NLP preprocessing pipeline

✔ Real-time deployment using Streamlit

✔ Confidence-based prediction

✔ Clean modular structure

✔ Logging support

---

##  Business Impact

- Reduces phishing and fraud risks
- Improves messaging platform security
- Automates spam filtering
- Enhances user trust

---

##  Author

**Hema Malini Gangumalla**
Aspiring Data Scientist | NLP & Deep Learning Enthusiast

📧 hemamalinig07@gmail.com

---

##  License

MIT License
