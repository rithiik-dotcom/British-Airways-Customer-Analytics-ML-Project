# ✈️ Maximizing Revenue by Predicting Customer Booking Behavior

**British Airways – Customer Analytics (Machine Learning Project)**

---

## 📌 Project Overview
This project builds a **predictive machine learning model** to identify **high-value customers with strong booking intent**.  
By predicting booking behavior accurately, the business can **target the right users**, maximize **revenue**, and improve **marketing ROI**.

---

## 📉 Business Problem
- **Imbalanced Data:**  
  Only **15%** of users complete a booking, making prediction highly imbalanced.

- **The Accuracy Trap:**  
  A baseline model achieved **85% accuracy** but was ineffective because it predicted *“No Booking”* for most users.
  - **Recall:** 14%
  - **Missed:** 86% of actual bookers

- **Objective:**  
  Shift focus from **accuracy** to **Recall and Precision**, prioritizing **revenue-generating customers**.

---

## 🛠️ Solution Strategy

### 1️⃣ Data Preparation & Feature Engineering
Intent-driven features were engineered to better capture customer behavior:
- **New Features**
  - Length of stay
  - Weekend flight indicator
  - Extras selected (meals, baggage)
  - Stay per flight hour
- **Feature Reduction**
  - Reduced features from **124 → 62** high-value features

---

### 2️⃣ Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)**
- Improved learning from minority-class (booking) customers

---

### 3️⃣ Model Selection
- **Baseline Model:** Random Forest  
- **Final Model:** **XGBoost**

**Why XGBoost?**
- Sequential error correction
- Strong performance on tabular data
- Robust handling of feature interactions

---

## 📊 Model Performance
The decision threshold was optimized to **0.30** to prioritize booking recall.

| Metric | Score | Business Impact |
|------|------|----------------|
| **ROC-AUC** | **0.786** | Good class separability |
| **Accuracy** | **80%** | Lower but more meaningful |
| **Recall (Booking)** | **53%** | Captures majority of bookers |
| **Precision** | **37%** | Acceptable trade-off for revenue |

> **Key Insight:**  
> Lower accuracy was accepted to achieve higher **Recall**, since **missing a booker costs more** than targeting a non-booker.

---

## 🧠 Key Drivers of Booking (SHAP Analysis)
SHAP analysis revealed the most influential factors:

1. **Booking Origin**  
   Higher intent from **Malaysia, China, and Australia**
2. **Length of Stay**  
   Longer stays increase booking probability
3. **Flight Timing**  
   Weekend flights and specific departure days matter
4. **Extras Selected**  
   Baggage and meal selection strongly indicate intent

---

## 🚀 Business Impact
This project converted an **accurate-but-useless model** into a **revenue-focused engine**.

- **Targeted Marketing:** Focus on top **30% high-probability users**
- **Revenue Protection:** Fewer missed high-value customers
- **Higher ROI:** Marketing spend aligned with true intent

---

## ▶️ How to Run the Project
```bash
pip install -r requirements.txt
python train_model.py
