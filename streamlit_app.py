import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(
    page_title="IOT - Failure Prediction", page_icon="⚙️", layout="wide"
)

st.title("⚙️ IOT Device Failure Prediction System")
st.write(
    "Interactive Machine Learning Application to forecast 7-day IoT device failures using real-time sensor metrics."
)


# Load and Preprocess Dataset
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("IOT_Devices.csv")

    # Map device types
    device_map = {
        "GPS_LOCK": "JT700",
        "IOT_SENSOR": "508N",
        "TRACKER": "310P",
        "TEMPERATURE_SENSOR": "400",
    }
    df["device_type"] = df["device_type"].map(device_map)

    # Drop identifiers
    columns_to_drop = [
        "record_id",
        "device_id",
        "installation_date",
        "record_date",
    ]
    df_clean = df.drop(columns=columns_to_drop)

    # One-Hot Encode
    categorical_cols = [
        "device_type",
        "location",
        "firmware_version",
        "device_status",
    ]
    df_encoded = pd.get_dummies(
        df_clean, columns=categorical_cols, drop_first=True
    )

    return df_encoded


# Train Models Function
@st.cache_resource
def train_models(df_encoded):
    X = df_encoded.drop(columns=["failure_next_7_days"])
    y = df_encoded["failure_next_7_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    trained_models = {}
    metrics = []

    for name, model in models.items():
        if name in ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)

        trained_models[name] = model
        metrics.append(
            {
                "Model": name,
                "Accuracy": f"{acc:.2%}",
                "F1-Score": f"{f1:.2f}",
                "Confusion_Matrix": cm,
            }
        )

    return trained_models, scaler, X.columns, pd.DataFrame(metrics)


# Execution
try:
    df_encoded = load_and_preprocess_data()
    models, scaler, feature_columns, metrics_df = train_models(df_encoded)

    # Main Layout Tabs
    tab1, tab2 = st.tabs(
        ["🔮 Real-Time Prediction", "📊 Model Performance & Benchmarks"]
    )

    with tab1:
        st.subheader("Input Device Metrics")
        col1, col2 = st.columns(2)

        # Generate form inputs dynamically or via default feature selection
        with col1:
            selected_model_name = st.selectbox("Select Prediction Model", list(models.keys()))
            
            # Numeric Feature Inputs
            temp = st.number_input("Operating Temperature (°C)", value=45.0, step=0.5)
            vibration = st.number_input("Vibration Level", value=0.02, step=0.001, format="%.3f")
            humidity = st.number_input("Humidity (%)", value=55.0, step=1.0)

        with col2:
            battery = st.slider("Battery Level (%)", 0, 100, 85)
            signal_strength = st.slider("Signal Strength (dBm)", -120, -30, -70)
            
            device_type = st.selectbox("Device Type", ["JT700", "508N", "310P", "400"])

        if st.button("Predict Failure Risk", type="primary"):
            # Construct Input DataFrame matching feature set
            input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

            # Assign numeric features if present in dataset
            numeric_mappings = {
                "temperature": temp,
                "vibration": vibration,
                "humidity": humidity,
                "battery_level": battery,
                "signal_strength": signal_strength,
            }

            for col, val in numeric_mappings.items():
                if col in input_data.columns:
                    input_data.at[0, col] = val

            # Set dummy encoding for selected device_type
            dt_col = f"device_type_{device_type}"
            if dt_col in input_data.columns:
                input_data.at[0, dt_col] = 1

            # Transform and Predict
            selected_model = models[selected_model_name]
            if selected_model_name in ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"]:
                input_scaled = scaler.transform(input_data)
                prediction = selected_model.predict(input_scaled)[0]
            else:
                prediction = selected_model.predict(input_data)[0]

            # Display Result
            st.divider()
            if prediction == 1:
                st.error("⚠️ **High Failure Risk Detected**: Maintenance required within 7 days.")
            else:
                st.success("✅ **Normal Operation**: Device is unlikely to fail within 7 days.")

    with tab2:
        st.subheader("Algorithm Comparison")
        st.dataframe(metrics_df[["Model", "Accuracy", "F1-Score"]], use_container_width=True)

        st.subheader("Confusion Matrices")
        cols = st.columns(2)
        for idx, row in metrics_df.iterrows():
            with cols[idx % 2]:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(
                    row["Confusion_Matrix"],
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax,
                    cbar=False,
                    xticklabels=["No Fail", "Fail"],
                    yticklabels=["No Fail", "Fail"],
                )
                ax.set_title(f"{row['Model']}")
                st.pyplot(fig)

except FileNotFoundError:
    st.error("Dataset `IOT_Devices.csv` not found. Please ensure the file is in the root directory.")
