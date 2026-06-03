# 🌾 SPI Forecasting Framework

<div align="center">

**Deep Learning and Machine Learning Framework for Standardized Precipitation Index (SPI) Forecasting**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Overview

This framework provides a complete pipeline for **SPI (Standardized Precipitation Index)** forecasting using both **Deep Learning (ConvLSTM3D)** and **Classical Machine Learning (Random Forest & XGBoost)** approaches. The framework supports:

- Multi-step ahead forecasting (Q = 1, 3, 6, 9, 12 months)
- Multiple input window sizes (P = 3, 6, 9, 12 months)
- Geospatial map visualizations and GeoTIFF exports
- Comprehensive evaluation metrics (WI, RMSE, MAE)
- Automatic caching for reproducibility

---

## 🏗️ Project Structure

.
├── config.py # Central configuration
├── dataset.py # PyTorch Dataset for spatiotemporal data
├── data_preparation.py # Data preparation for classical models
├── model_convlstm3d.py # ConvLSTM3D architecture with delta prediction
├── model_classic.py # RF and XGBoost models
├── train_model.py # Training loop for ConvLSTM3D
├── metrics.py # Evaluation metrics (WI, RMSE, MAE)
├── utils_data.py # Data loading, SPI calculation, caching
├── plots.py # Visualization utilities
├── visualization_spi.py # Map generation utilities
├── main.py # Main experiment runner
├── generate_monthly_maps.py # Monthly prediction maps (2025 test period)

---

## 🚀 Quick Start

```python
# 1. Set your data path
DATA_PATH = "data/pr_Area1.xlsx"

# 2. Run experiments
python main.py

This will:

Calculate SPI for the configured scale (default: 3 months)
Train all models for all (P, Q) combinations
Save results, metrics, and visualizations

# 3. Generate monthly prediction maps (2025)
python generate_monthly_maps.py
