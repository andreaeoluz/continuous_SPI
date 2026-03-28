# 🌾 SPI Forecasting Framework

**Multi-horizon drought forecasting** using deep learning and classical machine learning.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 What is this?

A complete experimental framework for **predicting drought conditions** using the Standardized Precipitation Index (SPI). The framework combines:

- **ConvLSTM3D** with attention mechanisms (temporal, spatial, channel)
- **Random Forest** and **XGBoost** with autoregressive forecasting
- Leakage-free SPI computation
- Systematic evaluation across multiple time horizons

---

## 🚀 Quick Start

```python
# 1. Set your data path
DATA_PATH = "data/pr_Area1.xlsx"

# 2. Run experiments
python main.py

# Generate the monthly maps
generate_monthly_maps.py