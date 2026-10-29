# QM 640: Data Analytics Capstone — Group 7
## Predictive Analytics and Multi-Agent Systems for Supply Chain Optimization in the Indian E-Commerce Sector: A Cluster-Based Machine Learning and Multi-Agent Simulation Framework for Heterogeneous B2C Retail Operations

**Author:** Subhashree.R (Group 7)  
**Institution:** Department of Decision Sciences, Walsh College  
**Course:** QM 640: Data Analytics Capstone  
**Mentor / Evaluator:** Mr. Sharath Srivatsa  
**Academic Term:** Fall 2026 Term  
**Interim Report Date:** October 29, 2026  

---

### Project Overview
This repository contains the end-to-end data analytics pipeline, exploratory data analysis (EDA), customer segmentation models, supervised predictive ensembles, and reinforcement learning simulation environment for Group 7's capstone project investigating supply chain volatility, delivery delays, and Return-to-Origin (RTO) cascades across Indian Tier-1, Tier-2, and Tier-3 cities.

Addressing faculty mentorship guidance on consumer non-homogeneity, this study implements **K-Prototypes clustering ($k = 4$, $S = 0.63$)** to decouple heterogeneous consumer cohorts before evaluating predictive and prescriptive supply chain models.

---

### Repository Structure
```text
QM640-ECommerce-Capstone-Group7/
├── data/
│   ├── raw/                  # Raw multi-channel transactions (N_raw = 131,420)
│   └── processed/            # Validated analytical dataset (N = 128,976) & demand data
├── notebooks/                # Analysis notebooks for EDA, clustering, and modeling
├── src/                      # Modular scripts for preprocessing and model evaluation
├── Interim_analysis_pipeline.py
├── requirements.txt
└── README.md
