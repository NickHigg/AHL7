import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Title
st.title("🏡 Ames Housing Price Predictor")
st.markdown("Predict the selling price of homes in Ames, Iowa using machine learning.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("AmesHousing.csv")  # <-- Updated to CSV
    return df

df = load_data()

# Basic Preprocessing: Drop columns with too many missing values
df.dropna(thresh=len(df) * 0.9, axis=1, inplace=True)
df.drop(columns=["Order", "PID"], inplace=True)

# Drop rows with any remaining NaNs
df.dropna(inplace=True)

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Streamlit User Input Form ---
st.sidebar.header("🏠 Input Home Features")

# Use most influential features for input
important_features = ["Gr Liv Area", "Overall Qual", "Year Built", "Total Bsmt SF", "Garage Area"]

user_input = {}
for feature in important_features:
    min_val = int(df[feature].min())
    max_val = int(df[feature].max())
    default_val = int(df[feature].median())
    user_input[feature] = st.sidebar.slider(f"{feature}", min_val, max_val, default_val)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Align input_df with X (for missing dummy columns)
input_df_full = pd.DataFrame(columns=X.columns)
for col in input_df.columns:
    input_df_full[col] = input_df[col]
input_df_full.fillna(0, inplace=True)

# Predict and display
if st.button("Predict Price"):
    prediction = model.predict(input_df_full)[0]
    st.success(f"💰 Estimated Sale Price: **${prediction:,.2f}**")
