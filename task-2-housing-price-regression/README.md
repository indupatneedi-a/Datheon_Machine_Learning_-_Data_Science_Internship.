# Task 2: Real Estate Housing Price Prediction (OLS, Ridge and Lasso)

Predicts median house values on the California Housing dataset and compares
Ordinary Least Squares (OLS), Ridge (L2) and Lasso (L1) regression.

## Dataset
California Housing: 20,640 districts, 10 columns, loaded from:
https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv

- Target: `median_house_value` (mean about 206,856, capped at 500,001)
- Missing values: 207 in `total_bedrooms`, filled with the median
- Categorical feature: `ocean_proximity` (one-hot encoded)

## Approach
1. Loaded the data with Pandas and checked missing values.
2. Engineered four interaction/ratio features: `rooms_per_household`,
   `bedrooms_per_room`, `population_per_household` and `income_x_rooms`.
3. Plotted a correlation heatmap to inspect multicollinearity.
4. Split the data 80/20 (16,512 training rows, 4,128 test rows) using 13 input features.
5. Imputed missing values (median), standardized numeric features and one-hot
   encoded the categorical feature inside a single scikit-learn pipeline.
6. Trained OLS, Ridge and Lasso. Tuned the alpha of Ridge and Lasso with
   GridSearchCV (5-fold cross-validation, alphas from 0.01 to 1000).
7. Compared the models using RMSE, MAE and R2, and plotted predicted vs actual
   values and residual distributions.

## Multicollinearity findings
Several features are very strongly correlated with each other:
- `total_bedrooms` and `households` (r = 0.98)
- `total_rooms` and `total_bedrooms` (r = 0.93)
- `longitude` and `latitude` (r = 0.92)
- `total_rooms` and `households` (r = 0.92)
- `population` and `households` (r = 0.91)

This is why regularized models (Ridge and Lasso) are worth comparing against
plain OLS: they reduce the instability that correlated features cause.

## Results
The full metrics table (RMSE, MAE and R2 for OLS, Ridge and Lasso), the best
alpha values, the predicted-vs-actual plots and the residual distribution plots
are in the notebook `Housing_Price_Regression.ipynb`.

## Tools
Python, Pandas, NumPy, Scikit-Learn (LinearRegression, Ridge, Lasso,
GridSearchCV), Seaborn, Matplotlib.

## How to run
1. Open the notebook in JupyterLab or JupyterLite.
2. Install the libraries: `pip install -r requirements.txt`
3. Run all cells from top to bottom.

## Files
https://jupyter.org/try-jupyter/lab/index.html?path=notebooks%2FHousing_Price_Regression.ipynb

## Screenshots
<img width="1751" height="703" alt="Screenshot_26-11-2026_125835_jupyter org" src="https://github.com/user-attachments/assets/b2ec3247-77f5-408b-b9b3-adb2e25dc3fa" />
<img width="1805" height="705" alt="Screenshot_26-11-2026_12581_jupyter org" src="https://github.com/user-attachments/assets/356a88cd-ed1a-424a-b3df-58bf0ecbca1c" />
![Uploading Screenshot_26-11-2026_125739_jupyter.org.jpeg…]()
