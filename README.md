# SPI Forecasting Framework

Experimental framework for **multi-horizon forecasting of the Standardized Precipitation Index (SPI)** using classical machine learning models and deep learning.

This repository contains the scripts used to conduct experiments for **spatio-temporal drought forecasting**.

---

## 🌍 Overview

Drought prediction is essential for water resource management and climate risk mitigation.  
This framework enables systematic experiments for **SPI forecasting across multiple temporal configurations**, comparing classical machine learning and deep learning approaches.

Key features:

- Multi-horizon SPI forecasting  
- Flexible historical input windows  
- Leakage-safe SPI computation  
- Comparison between classical ML and deep learning models  
- Spatial evaluation and visualization of predictions  

---

## ⚙️ Experimental Configuration

The framework evaluates combinations of:

**Historical window size**

P = {3, 6, 9, 12, 15, 18, 21, 24} months

**Forecast horizons**

Q = {1, 3, 6, 9, 12} months

Total experiments: 40 configurations


---

## 🧠 Implemented Models

### Deep Learning

- **ConvLSTM3D**
- Temporal attention
- Spatial attention
- Channel attention

### Classical Machine Learning

- **Random Forest**
- **XGBoost**

Classical models generate **multi-horizon forecasts using an autoregressive strategy**.

---

## 📁 Repository Structure
SPI-Forecasting-Framework
│
├── main.py # Main experimental pipeline
│
├── dataset.py # Dataset for SPI forecasting
│
├── data_preparation.py # Dataset preparation and sampling
│
├── utils_data.py # SPI computation (no data leakage)
│
├── model_convlstm3d.py # ConvLSTM3D architecture
│
├── model_classic.py # Random Forest and XGBoost models
│
├── train_model.py # Training and evaluation routines
│
├── visualization_spi_classes.py # Spatial visualization and metrics
│
├── data/
│ └── precipitation datasets
│
└── EXPERIMENTS/
└── saved models, metrics and figures


---

## 📊 Evaluation Metrics

Model performance is evaluated using:

- **Willmott Index (WI)**
- **Root Mean Square Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Nash–Sutcliffe Efficiency (NSE)**
- **Bias**

The framework also supports **SPI categorical evaluation** using WMO drought classes.

---

## 🚀 Running the Framework

Edit the dataset path in `main.py`:

```python
DATA_PATH = "data/pr_Area1.xlsx"

Run the experiments: python main.py

📈 Output

Results are stored in: EXPERIMENTS/