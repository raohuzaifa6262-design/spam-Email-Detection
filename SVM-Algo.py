import kagglehub
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# 📥 Download dataset from Kaggle
path = kagglehub.dataset_download("ozlerhakan/spam-or-not-spam-dataset")

print("Dataset path:", path)

# 📂 Find CSV file
files = os.listdir(path)
print("Files:", files)

# Load dataset (usually named spam_or_not_spam.csv)
file_path = os.path.join(path, files[0])
data = pd.read_csv(file_path)

print(data.head())

# 🧹 Clean data
data = data.dropna()

# Columns in this dataset:
# 'email' → message
# 'label' → 0 (not spam), 1 (spam)

X = data["email"]
y = data["label"]

# 🔢 Convert text → numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.2, random_state=42
)

# 🤖 Train SVM
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# 📊 Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 💾 Save model
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained & saved successfully!")