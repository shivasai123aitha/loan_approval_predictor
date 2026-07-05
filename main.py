import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("train.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# ==========================
# Handle Missing Values
# ==========================
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(
    df['Loan_Amount_Term'].median()
)
df['Credit_History'] = df['Credit_History'].fillna(
    df['Credit_History'].mode()[0]
)

# ==========================
# Drop Loan_ID
# ==========================
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# ==========================
# Encode Categorical Columns
# ==========================
encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "encoders.pkl")

print("\nData Types After Encoding:")
print(df.dtypes)

# ==========================
# Features and Target
# ==========================
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# ==========================
# Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Feature Scaling
# ==========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Train Model
# ==========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Predictions
# ==========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================
# Save Files
# ==========================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")
print("Files created:")
print("1. model.pkl")
print("2. scaler.pkl")
print("3. encoders.pkl")