# Sentiment Analysis of Text Reviews

This is a full-stack web application that classifies movie review sentiment in real-time. It demonstrates the complete machine learning lifecycle, from data cleaning and model training to creating a backend API and an interactive frontend UI.

---

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8fc330f2-397a-48ff-9641-a1da4bda9f8f" />

---

## 📖 Detailed Project Description

This project tackles the classic NLP problem of sentiment analysis. It begins with a raw 50,000-review dataset from IMDb. This data is rigorously cleaned and processed using NLTK.

Two distinct models are trained to compare different approaches:
1.  A **Baseline Model** using TF-IDF and Logistic Regression, which is fast and efficient.
2.  An **Advanced Model** using an LSTM (Long Short-Term Memory) neural network, which is designed to understand the context and sequence of words.

The best-performing model is then served via a **Flask REST API**, which handles raw text input, preprocesses it, and returns a JSON prediction. Finally, a **Streamlit UI** provides a user-friendly web interface that communicates with the Flask API, allowing anyone to enter a review and get a live sentiment classification.

## ✨ Features

* **Dual-Model Architecture:** Includes a simple `Logistic Regression` baseline (TF-IDF) and a more advanced `LSTM` deep learning model (TensorFlow/Keras) for contextual understanding.
* **REST API:** A robust `Flask` backend serves the trained models, complete with input validation and error handling.
* **Interactive UI:** A user-friendly `Streamlit` frontend to interact with the API in real-time.
* **Complete ML Pipeline:** Demonstrates data preprocessing (NLTK), feature engineering, training, and evaluation.
* **Decoupled Architecture:** The frontend (Streamlit) and backend (Flask) are separate applications, which is a modern, scalable design.

---

## 🤖 Model Overview

This project intentionally builds two different models to compare their performance and complexity.

### 1. Baseline: Logistic Regression (TF-IDF)

* **Technique:** Uses `TfidfVectorizer` (a "bag-of-words" approach) to convert text into numerical features based on word frequency and importance.
* **Pros:** Extremely fast, efficient, and highly interpretable. Provides a great performance baseline (achieved ~88-89% accuracy).
* **Cons:** Fails to understand context or word order (e.g., "not good" is seen as "not" + "good," which can be confusing for the model).

### 2. Advanced: LSTM (Deep Learning)

* **Technique:** A Recurrent Neural Network (RNN) that reads text sequentially, word by word. It uses `Keras Tokenizer` and `Embedding` layers to learn the contextual meaning of words.
* **Pros:** Can understand negation ("**not** bad") and the sequence of words, leading to a more nuanced understanding of language.
* **Cons:** Much slower to train and more computationally expensive.

The Flask API can be configured to serve either model by changing which files it loads.

---

## 🛠️ Tech Stack & Requirements

This project is built entirely in Python. The "complete requirements" are listed in the `requirements.txt` file.

* **Data Science:** Pandas, NumPy, NLTK, Scikit-learn
* **Deep Learning:** TensorFlow (Keras), h5py
* **Backend API:** Flask
* **Frontend UI:** Streamlit, Requests
* **Development:** Jupyter Notebooks

---
