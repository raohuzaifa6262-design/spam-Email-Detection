# Spam Email Detection (SVM) 📧

This is a Machine Learning project that classifies emails and messages as **Spam** or **Not Spam** using a Support Vector Machine (SVM) and TF-IDF Vectorization. The project includes a web application built with Streamlit for easy interaction.

## 🚀 Setup & Installation

If you've cloned this project and want to run it on your local machine, follow these steps to avoid any `ModuleNotFoundError` or dependency issues:

### 1. Install Dependencies
Open your terminal/command prompt in the project folder and install all the required packages by running:
```bash
pip install -r requirements.txt
```
*(This installs `scikit-learn`, `streamlit`, `pandas`, and `kagglehub`)*

### 2. Run the Application
Once the installation is complete, start the Streamlit web app by running:
```bash
python -m streamlit run streamlit-svm.py
```
*(Using `python -m` ensures Streamlit uses the exact environment where you just installed the dependencies).*

## 🧠 Model Training
If you want to retrain the model from scratch:
1. Run `python SVM-Algo.py`
2. It will automatically download the dataset from Kaggle, train the Linear SVM model, and save the updated `spam_model.pkl` and `vectorizer.pkl` files to the directory.
