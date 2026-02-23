# 🚀 Machine Learning Internship Projects  
### Internship Organization: MyDaily Work

This repository contains Machine Learning and NLP projects completed during my internship at **MyDaily Work**.

These projects demonstrate practical implementation of ML pipelines, model evaluation, and interactive deployment using Streamlit.

---

## 👤 About Me



## 👤 About Me

**Samsul Mondal**  
B.Tech in Computer Science & Engineering (2023–2027)  
Focused on Machine Learning,Deep Learning, NLP, and Scalable AI Systems 






# Movie Genre Classification System

A multi-model NLP application that predicts movie genres from title and plot summary using:

-  TF-IDF + Logistic Regression (Fast Inference)
-  DeBERTa Transformer Model (High Accuracy)

Built with Streamlit for interactive web-based predictions.

---

## Live Demo
(Add your deployed link here if available)

---

## Project Overview

This project solves a **multi-class movie genre classification** problem with **27 different genres**.

Since the dataset is highly imbalanced, the model is evaluated using:

> **Macro F1-Score** — suitable for imbalanced multi-class problems.

Users can select between:
- Fast lightweight model
-  High-accuracy transformer model

---

##  Models Used

### 1️. TF-IDF + Logistic Regression
- Feature extraction using TF-IDF
- Logistic Regression classifier
- Fast inference (CPU-friendly)

### 2️. DeBERTa Transformer
- Fine-tuned transformer model
- Better semantic understanding
- GPU-supported inference
- Higher Macro F1-score

---

##  Tech Stack

- Python
- Streamlit
- Scikit-learn
- HuggingFace Transformers
- PyTorch
- Pandas
- Joblib

---

##  Project Structure
movie_genre_prediction/
│
├── distilbart/
├── TFID_vector/
├── main.py
├── requirements.txt
├── README.md


---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/deathsamsul/MYDAILYWORK.git
cd movie-genre-classification

python -m venv genre
source genre/bin/activate  # Mac/Linux
genre\Scripts\activate     # Windows
pip install -r requirements.txt
streamlit run main.py
