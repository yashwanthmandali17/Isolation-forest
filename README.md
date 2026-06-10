# 🛡️ Traffic Flow Anomaly Detection using Isolation Forest

This project implements an unsupervised anomaly detection system for network traffic data using the Isolation Forest algorithm, deployed as an interactive Streamlit web application.

The system detects abnormal or potentially malicious traffic patterns by analyzing statistical and temporal network features.

---

## 🚀 Features

- Unsupervised anomaly detection using Isolation Forest
- Adjustable model parameters (contamination rate, number of trees)
- CSV upload support and built-in sample dataset
- 2D and 3D data visualizations
- Feature-wise statistical comparison
- Export results and summary reports
- Deployed using Streamlit Cloud

---

## 🧠 Methodology

1. Load network traffic dataset (CSV)
2. Remove label column if present
3. Convert boolean features to numeric values
4. Handle missing values
5. Scale features using StandardScaler
6. Train Isolation Forest model
7. Predict anomalies (-1 = anomaly, 1 = normal)
8. Visualize and export results

---

## 📊 Dataset

- Embedded System Network Security Dataset
- Contains network traffic features such as:
  - Packet size
  - Inter-arrival time
  - Packet count
  - Protocol and entropy-based features

The dataset is included for demonstration purposes.

---

## 🖥️ Streamlit Application

The application provides:
- Interactive parameter tuning
- Overview metrics
- 2D and 3D anomaly visualizations
- Statistical feature analysis
- CSV and report export options

---

## 📦 Installation & Local Setup

Clone the repository and install dependencies:
```bash
git clone https://github.com/Chandra-16/anomaly-detection.git

cd anomaly-detection
pip install -r requirements.txt
streamlit run Streamlit_app.py
```

---

## ☁️ Deployment

This project is deployed using Streamlit Cloud.

Steps:
1. Push code to GitHub
2. Add requirements.txt
3. Select Streamlit_app.py as the main file
4. Deploy from Streamlit Cloud dashboard

---

## 📁 Project Structure

anomaly-detection/
├── Streamlit_app.py
├── embedded_system_network_security_dataset.csv
├── requirements.txt
├── README.md
└── .gitignore


---

## 🧪 Technologies Used

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Plotly

---

## 🎯 Use Cases

- Network traffic anomaly detection
- Intrusion detection systems
- Embedded system security analysis
- Academic and research demonstrations

---
"# anomaly-detection" 
