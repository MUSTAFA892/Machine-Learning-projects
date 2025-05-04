

# 🇷🇺 Russian Car Plate Price Prediction 🏎️💰

## 📌 Overview

This project is part of a Kaggle competition aimed at building machine learning models to **predict the price of Russian vehicle registration plates**. The prices are based on historical sale offer data collected between 2021 and 2025. A plate’s value can depend on factors such as symbolic meaning, rarity, government affiliation, and market trends.

## 📅 Competition Timeline

* **Start**: 2 months ago
* **Close**: 1 month to go
* **Prizes**:
  🥇 1st Place – \$25
  🥈 2nd Place – \$15
  🥉 3rd Place – \$10
  *(Prizes are symbolic and non-commercial)*

## 📂 Dataset Description

The competition dataset includes:

* `train.csv`: Historical listings of car plates with price and date info.
* `test.csv`: Listings for which the price needs to be predicted.
* `sample_submission.csv`: Format for prediction submission.
* `supplemental_russian.py` & `supplemental_english.py`: Helpful resources for understanding region codes and identifying special plates.

Key components:

* **REGION\_CODES**: Maps Russian regions to numeric codes (e.g., "77" for Moscow).
* **GOVERNMENT\_CODES**: Lists government-affiliated or special-use plates.

## 🎯 Objective

Accurately predict the price of car plates in the test dataset. This includes understanding:

* Plate symbolism
* Government-associated numbers
* Regional significance
* Historical price trends

## 📊 Evaluation Metric

Submissions are scored using **Symmetric Mean Absolute Percentage Error (SMAPE)**:

* SMAPE accounts for both overestimates and underestimates.
* Lower SMAPE indicates better model performance.

## 📝 Submission Format

Submit predictions in CSV format:

```
id,price
2,100000
5,200000
6,300000
...
```

## 🤖 Our Approach

We developed and deployed a machine learning model that powers a **Telegram bot**, called:

> **AI оценка номера авто**
> It provides instant evaluations of car plate prices based on our model.

## 🧠 Key Modeling Ideas

* Feature engineering using supplemental region and government codes.
* Time-based price trend analysis.
* Plate pattern recognition (e.g., repeated digits or prestigious sequences).
* Price normalization and scaling.

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies
3. Explore the dataset and start experimenting with models!
4. Use the supplemental Python files to enrich your features.

## 📁 Files Included

| File                      | Description                         |
| ------------------------- | ----------------------------------- |
| `train.csv`               | Training data with known prices     |
| `test.csv`                | Test data for prediction            |
| `sample_submission.csv`   | Submission format                   |
| `supplemental_russian.py` | Original feature mappings           |
| `supplemental_english.py` | English-translated feature mappings |

## 📜 License

Usage of the dataset and participation in the competition is subject to **Kaggle Competition Rules**. This competition is for learning and research purposes only.

---
