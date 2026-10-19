# QM 640: Data Analytics Capstone — Group 7
## Predictive Analytics and Multi-Agent Systems for Supply Chain Optimization in the Indian E-Commerce Sector

**Author:** Subhashree.R  
**Institution:** Walsh College  
**Course:** QM 640: Data Analytics Capstone  
**Instructor:** Mr. Sharath Srivatsa  
**Academic Term:** Fall 2026 Term  

---

### Project Overview
This repository contains the end-to-end data science pipeline, exploratory data analysis (EDA), and predictive modeling codebase for Group 7's capstone project investigating supply chain volatility, delivery delays, and Cash-on-Delivery (COD) return cascades across Indian Tier-1, Tier-2, and Tier-3 cities.

### Key Interim Modeling Results
* **RQ1 (Disruption-Aware Delays):** **XGBoost Regressor** achieved a Holdout Test **RMSE of 2.98 hours** ($R^2 = 0.913$), reducing error by 35.6% below Multiple Linear Regression (OLS).
* **RQ2 (High-Risk Return Cascades):** **LightGBM + SMOTE** achieved a Holdout Test **Recall of 0.836** and **ROC-AUC of 0.884** (vs. 0.534 Recall and 0.742 AUC for baseline Logistic Regression).
* **RQ3 (Multi-Tier Inventory Volatility):** **Facebook Prophet** reduced pooled demand forecasting **MAPE to 6.84%** (a 51.6% error reduction below baseline ARIMA(2,1,2)).
* **RQ4 (Dynamic Adversarial Routing):** **PyTorch Deep Q-Network (DQN)** reduced mean last-mile delivery transit duration from **68.40 minutes** (Static Dijkstra) to **49.20 minutes** (a 28.1% transit time reduction, $p < 0.0001$).

---

### Dataset Provenance
* **Dataset:** E-Commerce Sales & Logistics Operations Dataset ($N = 128,976$ validated records derived from $131,420$ raw operational logs).
* **Source:** [Kaggle E-Commerce Sales Dataset](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data).

---

### Quickstart & Reproducibility
```bash
git clone [https://github.com/subhashreerajagopal/QM640-ECommerce-Capstone-Group7.git](https://github.com/subhashreerajagopal/QM640-ECommerce-Capstone-Group7.git)
cd QM640-ECommerce-Capstone-Group7
pip install -r requirements.txt
python interim_analysis_pipeline.py
