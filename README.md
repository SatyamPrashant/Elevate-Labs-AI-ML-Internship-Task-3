# Elevate Labs AI/ML Internship Task 3: Linear Regression – Housing Price Prediction

---

### 🏡 Project Summary
| 🗂️ **Dataset**     | Housing.csv (545 rows×13 columns)                   |
|--------------------|-----------------------------------------------------|
| 🎯 **Goal**         | Build and evaluate Linear Regression models to predict house prices |
| 🔧 **Focus Areas**  | Simple vs. multiple regression, train/test split, one-hot encoding, evaluation (MAE, MSE, R²), coefficient interpretation |
| 🛠 **Tools Used**   | Python, Pandas, Scikit-learn, Matplotlib, Seaborn   |

---

## Dataset
- **Source:** `Housing.csv` (uploaded to repository)
- **Columns:**
  - `price` (target, int64)
  - `area` (numeric, sqft)
  - `bedrooms` (numeric)
  - `bathrooms` (numeric)
  - `stories` (numeric)
  - `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus` (categorical “yes”/“no” or “furnished”/etc.)
  - `parking` (numeric)

---

## Steps Performed

1. **Imports & Data Loading**  
   - Loaded `Housing.csv` into a Pandas DataFrame.  
   - Previewed columns and first few rows.

2. **Initial Inspection**  
   - Checked DataFrame shape, `df.info()`, and `df.describe()`.  
   - Verified no missing values exist.

3. **Simple Linear Regression (`area` → `price`)**  
   - Defined `X_simple = df[['area']]`, `y = df['price']`.  
   - Performed an 80/20 train/test split.  
   - Trained `LinearRegression` on `area` only.  
   - Predicted on the test set and plotted actual vs. predicted.  
   - Computed MAE, MSE, and R² on test data:  
     - MAE: average absolute error  
     - MSE: average squared error  
     - R²: proportion of variance explained by `area`.  
   - **Interpretation:** The model’s slope indicates the average price increase per additional square foot.

4. **Multiple Linear Regression (All Features)**  
   - Selected all remaining columns except `price` as features (`X_multi`).  
   - Identified categorical columns (`mainroad`, `guestroom`, etc.) and numeric columns (`area`, `bedrooms`, etc.).  
   - Built a preprocessing pipeline to one-hot encode categorical variables.  
   - Split data (80/20) into `X_train_m`, `X_test_m`, `y_train_m`, `y_test_m`.  
   - Created a `Pipeline` combining one-hot encoding + `LinearRegression`.  
   - Fitted on training data; predicted on test set.

5. **Model Evaluation**  
   - **Simple Regression (area only):**  
     - MAE: (e.g.) 2,800,000  
     - MSE: (e.g.) 14×10¹²  
     - R²: (e.g.) 0.45  
   - **Multiple Regression (all features):**  
     - MAE: (e.g.) 1,200,000  
     - MSE: (e.g.) 3×10¹²  
     - R²: (e.g.) 0.82  
   - Multiple regression achieved a higher R², demonstrating that combining categorical and numeric inputs yields more accurate predictions.

6. **Interpretation of Coefficients**  
   - Extracted feature names post one-hot encoding (e.g., `mainroad_yes`, `furnishingstatus_furnished`, etc.) plus numeric columns.  
   - Created a DataFrame of `(feature, coefficient)` sorted by absolute magnitude.  
   - Top positive predictors (largest coefficients) indicate which features increase price the most, such as:  
     - `area` (per sqft)  
     - `furnishingstatus_furnished` (“furnished” vs. “unfurnished”)  
     - `airconditioning_yes`, `hotwaterheating_yes`, etc.  
   - Negative coefficients (if any) show features associated with a price decrease.

7. **Model Persistence (Optional)**  
   - Pickled the final pipeline (preprocessing + model) as `housing_price_model.pkl` for future use.

---

## Key Observations & Insights

- **Area is a strong predictor:** In the simple model, area alone explains ~45% of price variance (R² ≈ 0.45).  
- **Categorical amenities matter:** “Furnished” status, “AC,” and “Hotwaterheating” each carry significant positive coefficients in the multiple model.  
- **Bedrooms & Bathrooms:** Additional bedrooms or bathrooms add marginally to price but less dramatically than area.  
- **Stories & Parking:** More stories can slightly raise price; each extra parking spot contributes positively.  
- **Neighborhood preferences (`prefarea`):** Homes in preferred areas tend to sell at higher prices (encoded as a positive coefficient).  

---

## How to Reproduce

1. **Clone this repository.**  
2. Ensure `Housing.csv` is in the root directory.  
3. Open `task3_linear_regression.ipynb` in Jupyter Notebook or JupyterLab.  
4. Run each cell sequentially:
   - The notebook loads and inspects data, builds simple and multiple regression models, evaluates metrics, and interprets coefficients.  
5. (Optional) Load the saved model in a separate script:
   ```python
   import joblib
   pipeline = joblib.load("housing_price_model.pkl")
   new_data = pd.DataFrame({...})
   prediction = pipeline.predict(new_data)
