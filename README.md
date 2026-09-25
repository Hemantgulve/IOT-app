# IoT Device Failure Prediction Pipeline

A machine learning pipeline designed to predict IoT device failures within a 7-day window using sensor metrics, device metadata, and classification algorithms.

##  Key Features
- **Data Preprocessing & Encoding**: One-Hot Encoding for categorical features (`device_type`, `location`, `firmware_version`) and standardized label mapping.
- **Data Leakage Prevention**: Feature scaling using `StandardScaler` fitted strictly on training splits.
- **Model Comparison**: Benchmarked Logistic Regression, KNN, Naive Bayes, and Decision Trees using Accuracy, Precision, Recall, and F1-Score.
- **Visual Diagnostics**: Evaluated model performance using Seaborn confusion matrices.

## 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | 0.92 | 0.89 | 0.91 | **0.90** |
| **Logistic Regression** | 0.87 | 0.84 | 0.85 | **0.84** |
| **KNN** | 0.85 | 0.82 | 0.83 | **0.82** |
| **Naive Bayes** | 0.79 | 0.75 | 0.78 | **0.76** |

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib

## 🚀 How to Run
```bash
git clone [https://github.com/your-username/iot-device-failure-prediction.git](https://github.com/your-username/iot-device-failure-prediction.git)
cd iot-device-failure-prediction
pip install -r requirements.txt
python src/train_models.py
