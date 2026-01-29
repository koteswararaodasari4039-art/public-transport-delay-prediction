# train_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("../data/transport_delay.csv")  # Replace with your CSV path

# -----------------------------
# 2. Define features and target
# -----------------------------
cat_cols = ["weather_condition", "event_type", "season"]
num_cols = ["temperature_C", "humidity_percent", "traffic_congestion_index", "peak_hour", "holiday"]
target = "delayed"

# -----------------------------
# 3. OneHotEncode categorical columns
# -----------------------------
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(df[cat_cols])

# Transform categorical columns
encoded_cat = encoder.transform(df[cat_cols])

# -----------------------------
# 4. Combine numeric and categorical features
# -----------------------------
X = np.hstack([encoded_cat, df[num_cols].values])
y = df[target].values  # 1D array ✅

# -----------------------------
# 5. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 6. Train RandomForestClassifier
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate the model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# 8. Save model and encoder
# -----------------------------
joblib.dump(model, "../models/delay_classifier.pkl")
joblib.dump(encoder, "../models/encoder.pkl")

print("✅ Model and encoder saved in 'models/' folder")
