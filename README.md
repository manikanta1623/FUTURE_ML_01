# 📊 Sales & Demand Forecasting using SARIMA (Machine Learning Project)

## 📌 Project Overview

This project focuses on forecasting future sales using historical time-series data. A SARIMA (Seasonal AutoRegressive Integrated Moving Average) model is used to capture both trend and seasonal patterns in the data, enabling accurate demand prediction.

---

## 🎯 Objective

* Predict future sales based on historical data
* Analyze trends and seasonal patterns
* Support better business decision-making

---

## 🛠️ Tools & Technologies Used

* Python
* Pandas
* Matplotlib
* Statsmodels (SARIMA)
* Scikit-learn

---

## 📂 Dataset

The dataset contains time-series sales data with:

* Date → Timestamp
* Sales → Sales values

Data is cleaned and formatted for time-series analysis.

---

## 🔍 Project Workflow

### 1. Data Preprocessing

* Converted Date column to datetime format
* Set Date as index
* Handled missing values
* Ensured daily frequency of data

---

### 2. Exploratory Data Analysis

* Visualized sales trends over time
* Identified seasonal patterns

---

### 3. Model Building (SARIMA)

* Applied SARIMA model to capture:

  * Trend
  * Seasonality

* Model Parameters:

  * Order: (1,1,1)
  * Seasonal Order: (1,1,1,7)

---

### 4. Model Evaluation

* Split data into training and testing sets
* Evaluated performance using:

  * Mean Absolute Error (MAE)

---

### 5. Forecasting

* Predicted sales on test data
* Generated future forecast for next 30 days

---

## 📊 Results

* The model successfully captured trend and seasonality
* Forecast values closely follow actual data
* Achieved reasonable accuracy

---

## 📈 Visualization

* Training data (blue line)
* Actual test data (green line)
* Forecasted values (red dashed line)

---

## 💡 Business Impact

This model helps businesses in:

* Inventory planning
* Demand forecasting
* Revenue estimation
* Data-driven decision making

---

## 📁 Project Structure

FUTURE_ML_01/

* sarima_forecast.ipynb
* clean_sales_data.csv
* forecast_graph.png
* README.md

---

## 🚀 Conclusion

SARIMA is an effective model for time-series forecasting, especially when data shows seasonal behavior. This project demonstrates how machine learning can be applied to real-world business problems.

---

## 🔗 Future Improvements

* Use Auto-SARIMA for parameter tuning
* Try advanced models like Prophet or LSTM
* Use larger real-world datasets

---

## 📈 Forecast Graph

![Forecast Graph](<img width="1368" height="723" alt="Screenshot 2026-04-26 112749" src="https://github.com/user-attachments/assets/3a0e62a0-a2e5-44af-91c9-a58081525960" />
)
## 🙌 Acknowledgement

This project is developed as part of a Machine Learning Internship to gain practical experience in time-series forecasting.

---
