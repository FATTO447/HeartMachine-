# Supervised Learning with scikit-learn  

This repository contains my notes, code, and practice exercises from completing the **Supervised Learning with scikit-learn** course on DataCamp (completed: **Aug 31, 2025**).  

---

## 📌 Course Highlights  
- Building and evaluating **supervised learning models**  
- Using **scikit-learn** to train models on real datasets  
- Techniques for **improving model performance**  
- Hands-on practice with **classification & regression**  

---

## 📂 Repository Structure  
- `notebooks/` → Jupyter notebooks with code experiments  
- `datasets/` → Sample datasets used for testing models  
- `models/` → Saved models & outputs from experiments  

---

## 🚀 Next Steps  
- Apply learned techniques on **real-world projects**  
- Explore **hyperparameter tuning** and **model evaluation metrics**  
- Extend into **unsupervised learning** and **deep learning**  

---

## 🛠️ Technologies Used  
- Python 🐍  
- scikit-learn  
- pandas, numpy, matplotlib  

---

## 💻 Example Code  

python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

🙌 Resource
Course: DataCamp -> https://app.datacamp.com/learn/courses/supervised-learning-with-scikit-learn

BY: FATMA BADAWY
