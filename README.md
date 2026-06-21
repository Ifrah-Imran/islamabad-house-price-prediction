Below is your **README in clean “edit-ready” format** (you can directly paste into GitHub and tweak names/links later).

# 🏠 Islamabad House Price Prediction System

A machine learning-powered web application that predicts residential property prices in Islamabad, Pakistan using real estate data scraped from Zameen.com.

The system applies multiple regression models and a structured preprocessing pipeline to estimate property values based on housing features.

---

## 📌 Project Overview

This project is built to address inconsistent and informal property pricing in the Pakistani real estate market by using a data-driven approach.

The model predicts house prices based on key features such as:
- Area
- Location
- Bedrooms
- Bathrooms
- Kitchens
- Drawing rooms
- Parking spaces
- Servant quarters
- Store rooms

A Streamlit web application allows users to input these features and receive real-time price predictions.

---

## ⚙️ Features

- House price prediction for Islamabad properties
- Interactive Streamlit web application
- Custom web scraping pipeline for data collection
- Data cleaning and preprocessing workflow
- Comparison of multiple regression models
- Location-aware feature handling
- Real-time prediction interface

---

## 📊 Dataset

The dataset was collected using a custom scraper from Zameen.com.

### Dataset Summary
- Total listings collected: ~400
- Final processed samples: 399
- Unique locations: 140+
- Training samples: 319
- Test samples: 80

---

### 📌 Features Used

- Area (Marla)
- Location (housing society / sector)
- Bedrooms
- Bathrooms
- Kitchens
- Drawing rooms
- Parking spaces
- Servant quarters
- Store rooms

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Removal of duplicate listings
- Missing value imputation using median values
- Log transformation of target variable (price)
- Location normalization and cleaning
- Label encoding for categorical variables
- Frequency encoding for high-cardinality locations
- Location tier grouping (Budget / Mid / Premium / Ultra)

---

## 🤖 Machine Learning Models

The following regression models were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- CatBoost Regressor

---

## 📈 Model Performance

| Model | R² Score | MAPE |
|------|----------|------|
| Linear Regression | 0.8888 | 30.96% |
| Decision Tree | 0.6258 | 37.81% |
| Random Forest | 0.7701 | 30.38% |
| Gradient Boosting | 0.8899 | 29.60% |
| XGBoost | 0.8023 | 29.57% |
| CatBoost | 0.6774 | 31.24% |
| Final Calibrated Model | **0.9007** | 31.09% |

---

## 🧠 Final Model

The deployed system uses a:

**Location-Calibrated Gradient Boosting Pipeline**

This combines:
- Gradient Boosting predictions
- Location-based median price adjustments

to improve prediction accuracy across different areas.

---

## 🌐 Web Application

Built using Streamlit, the application allows users to:

- Select property location
- Enter house features (area, bedrooms, etc.)
- Get real-time predicted price

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- BeautifulSoup
- Requests
- Streamlit

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
````

### 2. Create virtual environment (optional)

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 📌 Key Findings

* Gradient Boosting performed best among standalone models
* XGBoost showed strong performance in terms of error reduction
* Linear Regression performed well after log transformation
* Tree-based models were sensitive to overfitting
* Location-based calibration improved final prediction stability

---

## 🚀 Future Improvements

* Expand dataset to other cities in Pakistan
* Add geospatial features (latitude/longitude)
* Include proximity-based features (schools, hospitals, markets)
* Improve location encoding methods
* Deploy application on cloud platforms

---

```
