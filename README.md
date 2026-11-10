# QM 640 Data Analytics Capstone — Group 7
## Predictive Analytics and Multi-Agent Systems for Supply Chain Optimization in the Indian E-Commerce Sector
**Author:** Subhashree.R (Group 7)  
**Institution:** Department of Decision Sciences, Walsh College  
**Course:** QM 640: Data Analytics Capstone  
**Faculty Mentor:** Mr. Sharath Srivatsa  
**Term:** Fall 2026 Term  

[![Python 3.10](https://img.shields.io/badge/Python-3.10.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

### Project Overview
This repository contains the complete, reproducible analytical codebase, customer segmentation pipeline, predictive machine learning models, sequential time-series forecasting, and multi-agent reinforcement learning (MARL) simulation environment supporting the QM 640 Capstone Final Report.

The study investigates $N = 128,976$ validated order transactions from the Indian B2C e-commerce sector, resolving last-mile delivery delays, high-risk Return-to-Origin (RTO) cascades, multi-tier warehouse stockouts, and urban courier routing gridlock.

---

### Repository Architecture

```text
QM640-ECommerce-Capstone-Group7/
├── data/
│   ├── raw/
│   │   ├── Amazon_Sale_Report.csv               # Raw multi-channel B2C orders (N_raw = 131,420)
│   │   ├── Cloud_Warehouse_Compersion_Chart.csv # 3PL benchmark logs (Shiprocket vs. INCREFF)
│   │   ├── India_PIN_Centroid_Registry.csv      # Postal PIN centroid coordinates
│   │   └── README.md                            # Download instructions for raw datasets
│   └── processed/
│       ├── ecommerce_clean_clustered_N128976.parquet # N = 128,976 validated clustered records
│       └── warehouse_demand_N150.csv             # Bi-daily warehouse demand periods (N = 150)
├── notebooks/
│   ├── 01_data_hygiene_and_eda.ipynb            # Data cleaning, missingness, & EDA
│   ├── 02_kprototypes_segmentation.ipynb         # Mixed-type K-Prototypes clustering (k = 4)
│   ├── 03_rq1_rq2_supervised_models.ipynb       # XGBoost regression & SMOTE-LightGBM
│   └── 04_rq3_prophet_rq4_marl_dqn.ipynb        # Prophet forecasting & PyTorch DQN simulation
├── src/
│   ├── preprocessing.py                         # Stratified median imputation & anti-leakage splits
│   ├── clustering.py                            # K-Prototypes optimization & centroid extraction
│   ├── models.py                                # Model training, Bayesian tuning, & evaluation
│   ├── agents.py                                # PyTorch DQN multi-agent routing environment
│   └── api.py                                   # Production FastAPI REST microservice endpoints
├── docker/
│   ├── Dockerfile                               # Container specification for REST microservices
│   └── docker-compose.yml                       # Multi-container service deployment config
├── requirements.txt                             # Pinned library dependencies
└── README.md                                    # End-to-end execution and reproduction runbook
```

---

### Quick Start: Reproducing Results in Google Colab

Run the complete pipeline end-to-end in Google Colab with 100% deterministic reproducibility:

```python
# 1. Clone repository
!git clone https://github.com/subhashreerajagopal/QM640-ECommerce-Capstone-Group7.git
%cd QM640-ECommerce-Capstone-Group7

# 2. Install pinned dependencies
!pip install -q kmodes==0.12.2 xgboost==1.7.6 lightgbm==4.0.0 imbalanced-learn==0.10.1 prophet==1.1.4 fastapi==0.100.0 uvicorn==0.22.0

# 3. Execute automated pipeline
import os, random, numpy as np
SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)

from src.preprocessing import load_and_clean_data
from src.clustering import run_kprototypes_segmentation
from src.models import train_rq1_xgboost, train_rq2_lightgbm, run_rq3_prophet, run_rq4_marl_dqn

df = load_and_clean_data(random_state=SEED)
df_clustered, _ = run_kprototypes_segmentation(df, random_state=SEED)
rq1 = train_rq1_xgboost(df_clustered, random_state=SEED)
rq2 = train_rq2_lightgbm(df_clustered, random_state=SEED)
rq3 = run_rq3_prophet("data/processed/warehouse_demand_N150.csv", random_state=SEED)
rq4 = run_rq4_marl_dqn(episodes=300, test_episodes=64, random_state=SEED)

print(f"[SUCCESS] RQ1 Holdout RMSE: {rq1['test_rmse']:.2f} h (Baseline OLS: {rq1['ols_rmse']:.2f} h)")
print(f"[SUCCESS] RQ2 Holdout Recall: {rq2['test_recall']:.3f} | ROC-AUC: {rq2['test_auc']:.3f}")
print(f"[SUCCESS] RQ3 Out-of-Sample MAPE: {rq3['prophet_mape']:.2f}% (ARIMA: {rq3['arima_mape']:.2f}%)")
print(f"[SUCCESS] RQ4 MARL Mean Transit: {rq4['marl_min']:.2f} min (Dijkstra: {rq4['static_min']:.2f} min)")
```

---

### Key Empirical Findings

| Research Question | Task / Domain | Baseline Model | Proposed Model | Improvement | Hypothesis Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RQ1** | Disruption Delay Regression | OLS ($RMSE = 4.63	ext{ h}$) | **XGBoost ($RMSE = 2.98	ext{ h}$)** | **$-35.6\%$ Error** | Reject $H_{10}$ ($p < .0001$) |
| **RQ2** | RTO Cascade Classification | Logistic Regression ($AUC = 0.742$) | **SMOTE-LightGBM ($AUC = 0.884$)** | **$+56.6\%$ Recall ($0.836$)** | Reject $H_{20}$ ($p < .0001$) |
| **RQ3** | Warehouse Demand Forecasting | ARIMA ($MAPE = 14.12\%$) | **Prophet ($MAPE = 6.84\%$)** | **$-51.6\%$ Error** | Reject $H_{30}$ ($p < .0001$) |
| **RQ4** | Dynamic Urban Routing | Static Dijkstra ($68.40	ext{ min}$) | **PyTorch MARL DQN ($49.20	ext{ min}$)** | **$-28.1\%$ Transit Time** | Reject $H_{40}$ ($p < .0001$) |

---

### Docker Microservice Deployment

To spin up the containerized FastAPI REST microservice locally:

```bash
docker compose -f docker/docker-compose.yml up --build
```
API Documentation will be live at `http://localhost:8000/docs`.

---

### License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
