# Crop Cost and Yield Analysis using Machine Learning

This project focuses on analyzing agricultural data across Indian states and crops to identify cost trends, yield efficiencies, and profitability. It also includes a machine learning model (XGBoost Regressor) to predict **Cost of Production** based on other crop and regional features.

## 📌 Project Objectives

- Clean and transform consolidated crop datasets extracted from Excel files.
- Perform exploratory data analysis (EDA) to identify trends and validate hypotheses.
- Use regression models to predict the **Cost of Production (Rs./Qtl)**.
- Identify important features affecting crop cost and yield using SHAP explainability.
- Provide insights for farmers, policymakers, and agri-economists to improve decision-making.

---

## 📊 Key Features

- Extracts and transforms Excel-based datasets into clean, structured data.
- Validates hypotheses using visualizations (correlation, scatter plots, line charts).
- Trains a regression model using XGBoost.
- Visualizes feature importance using SHAP plots.
- Outputs predictions and model metrics (RMSE, R² Score).

---

## 🧠 Hypotheses Tested

1. **States with higher derived yield have lower cost per unit of production.**  
2. **Crops like wheat and rice show more stable profit margins across years.**

---

## 🧪 Tech Stack & Libraries

- **Python 3.8+**
- **pandas**, **numpy** for data manipulation
- **matplotlib**, **seaborn** for visualization
- **XGBoost** for regression modeling
- **scikit-learn** for metrics and model validation

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/crop-cost-prediction.git
```

### 2. Virtual Env setup

```bash
python -m venv venv
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install libraries
```bash
pip install -r requirements.txt
```

### 4. Run Backend
```bash
python manage.py runserver
```

### 5. Frontend setup
```bash
cd crop-ui
npm install
npm run start
```
## Frontend will be running on `localhost:3000`

## Refer `crops/notebook/CropData_Transformation.ipynb` for Data Transformation & Model creation details

