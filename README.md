# Loan Approval Predictor

## InternId :CITS6520

## 📌 Project Overview

The **Loan Approval Predictor** is a Machine Learning project that predicts whether a loan application will be **Approved** or **Rejected** based on an applicant's personal and financial information.

This project uses a **Random Forest Classifier** and provides a simple **Streamlit web application** for making predictions.

---

## 🚀 Features

* Data preprocessing and cleaning
* Handling missing values
* Encoding categorical variables
* Feature scaling using StandardScaler
* Loan approval prediction using Machine Learning
* Interactive web application using Streamlit
* Model and scaler persistence using `.pkl` files

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

---

## 📂 Project Structure

```text
Loan_Approval_Predictor/
│
├── train.csv
├── main.py
├── app.py
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── encoders.pkl
└── README.md
```

---

## 📊 Dataset

The project uses the **Loan Prediction Dataset** from Kaggle.

Dataset Link:
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

---

## 🔍 Input Features

* Gender
* Married
* Dependents
* Education
* Self Employed
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Loan_Approval_Predictor.git
cd Loan_Approval_Predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python main.py
```

This generates:

* model.pkl
* scaler.pkl
* encoders.pkl

---

## ▶️ Run the Web Application

```bash
python -m streamlit run app.py
```

---

## 📈 Model Performance

* Algorithm: Random Forest Classifier
* Accuracy: Approximately 75% – 85%

---

## 📸 Sample Prediction

### Loan Approved

```text
Gender: Male
Married: Yes
Dependents: 1
Education: Graduate
Self Employed: No
Applicant Income: 8000
Coapplicant Income: 2000
Loan Amount: 150
Loan Amount Term: 360
Credit History: 1
Property Area: Semiurban
```

Output:

```text
Loan Approved ✅
```

---

## 🎯 Future Improvements

* Use XGBoost for higher accuracy.
* Deploy on Streamlit Cloud.
* Add feature importance visualization.
* Integrate database support.

---


