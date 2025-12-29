# Maximizing Revenue by Predicting Customer Booking Behavior ✈️

**Project Context:** British Airways – Customer Analytics | [cite_start]ML Project [cite: 3]

## 📌 Project Overview
[cite_start]This project aims to build a predictive model that identifies high-value customers with a strong intent to book flights[cite: 7]. [cite_start]By accurately predicting booking behavior, the business can target specific user segments to maximize revenue and improve marketing ROI[cite: 7, 68].

## 📉 The Business Problem
* [cite_start]**Imbalanced Data:** Only **15%** of users typically complete a booking, making the prediction task highly imbalanced[cite: 5, 6].
* [cite_start]**The "Accuracy Trap":** A baseline model achieved **85% accuracy** but was practically useless because it predicted "No Booking" for almost everyone[cite: 9, 10]. [cite_start]It had a Recall of only **14%**, meaning it missed **86%** of actual bookers[cite: 11, 12].
* [cite_start]**Goal:** Shift focus from raw accuracy to **Recall and Precision**, prioritizing the identification of actual revenue-generating customers[cite: 13, 45].

## 🛠️ Solution Strategy

### 1. Data Preparation & Feature Engineering
[cite_start]We engineered intent-based features to capture customer behavior better[cite: 15]:
* [cite_start]**New Features:** Length of stay, Weekend flight indicators, "Extras" selected, and Stay per flight hour[cite: 16, 17, 18, 19].
* [cite_start]**Dimensionality Reduction:** Reduced the feature set from 124 to **62 high-value features**[cite: 21].

### 2. Handling Imbalance
* [cite_start]Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset and improve the model's ability to learn from booking customers[cite: 23].

### 3. Model Selection
* [cite_start]**Algorithm:** Upgraded from Random Forest to **XGBoost**[cite: 24].
* [cite_start]**Why XGBoost?** Chosen for its sequential error learning, strong performance on tabular data, and stability[cite: 25, 26, 27].

## 📊 Model Performance
[cite_start]By optimizing the decision threshold to **0.30**, we significantly improved the capture of high-value customers compared to the baseline[cite: 36].

| Metric | Score | Impact |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.786** | [cite_start]Good separability between classes[cite: 37]. |
| **Accuracy** | **80%** | [cite_start]Slightly lower than baseline, but more valuable[cite: 40]. |
| **Recall (Booking)** | **53%** | [cite_start]Significantly higher capture of actual bookers[cite: 42]. |
| **Precision** | **37%** | [cite_start]Acceptable trade-off for higher revenue capture[cite: 44]. |

> [cite_start]**Key Takeaway:** We accepted a lower overall accuracy to gain higher Recall, acknowledging that false negatives (missed customers) are costlier to the business than false positives[cite: 45, 53].

## 🧠 Key Drivers of Booking (SHAP Analysis)
[cite_start]Using SHAP values, we identified the top factors influencing a customer's decision to book[cite: 61]:
1.  [cite_start]**Booking Origin:** High intent from Malaysia, China, and Australia[cite: 56].
2.  [cite_start]**Length of Stay:** Longer stays correlate with higher booking probability[cite: 57, 64].
3.  [cite_start]**Flight Timing:** Weekend flights and specific departure days are strong indicators[cite: 58].
4.  [cite_start]**Extras:** Users selecting extra baggage or meals have a significantly higher likelihood of booking[cite: 59, 63].

## 🚀 Business Impact
[cite_start]This project successfully transformed an "accurate but useless" model into a revenue-focused engine[cite: 69].
* [cite_start]**Targeted Marketing:** The model allows the business to target the top **30%** of high-probability users[cite: 66].
* [cite_start]**Revenue Protection:** Drastically reduces the number of missed high-value customers[cite: 67].
* [cite_start]**Efficiency:** Improves marketing ROI by focusing resources on users with genuine booking intent[cite: 68].

---

### How to Run This Project
*(Add your specific installation steps here)*
