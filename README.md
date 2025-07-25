HDB Resale Price Prediction

This project uses machine learning to predict resale prices of HDB flats in Singapore, helping Cataria Property Solutions provide accurate price insights for prospective buyers, current owners, and real estate analysts.

Features





Data Preprocessing: Scales numerical features (e.g., floor_area_sqm), encodes categorical features (e.g., flat_type as ordinal, town_name as nominal).



Models: Linear Regression (baseline), Ridge, Lasso, and ElasticNet.



Hyperparameter Tuning: Optimizes Ridge and Lasso using GridSearchCV and RandomizedSearchCV.



Evaluation: Measures performance with Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R².



Configurable: Uses config.yaml to adjust data paths, model parameters, and tuning settings.



EDA: Explores correlations and feature importance (e.g., storey_range as ordinal).

Problem Synopsis

Problem: How can we predict HDB resale prices accurately?
Additional Questions:





Requirements





Python 3.11.11



Libraries:

pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
pyyaml==6.0.1


Installation





Install Python: Download Python 3.11.11 from python.org.

Install Libraries:

pip install -r requirements.txt



Explore EDA: Open eda.ipynb in Jupyter Notebook to view data analysis (e.g., correlations like floor_area_sqm 0.71).



Check Config: Edit config.yaml to adjust:


Data path: data.file (e.g., data/resale_transaction.csv).



Target: data.target (set to resale_price).



Models: Enable/disable Linear, Ridge, Lasso, ElasticNet.



Tuning: Modify alpha values (e.g., [0.1, 1, 10]) or max_iter for Lasso.

Run the Pipeline:

chmod +x run.sh
./run.sh

This executes main.py, which:





Loads and preprocesses data.



Trains models (Linear, Ridge, Lasso, ElasticNet).



Tunes Ridge and Lasso.



Saves models to outputs/models/ and plots to outputs/plots/.



View Results: Check the terminal for metrics (e.g., MAE, R²) and outputs/plots/ for coefficient plots.

Data Overview

The dataset (resale_transaction.csv) contains 84,465 HDB resale transactions with features:





resale_price: Target variable (mean $438,679.30).



floor_area_sqm: Flat size in square meters (mean 97.52, Spearman 0.71).



flat_type: Flat type (e.g., 3 ROOM, ordinal: 1 ROOM = 0, EXECUTIVE = 6).



storey_range: Floor level (e.g., 01-03, converted to numbers, ordinal).



town_name: Town (e.g., ANG MO KIO, nominal).



flatm_name: Flat model (e.g., Improved, nominal).



month, year: Sale date (numerical).



remaining_lease_months: Lease left in months (numerical).



Dropped Columns: id, block, street_name, lease_commence_date, remaining_lease (too specific or redundant).



Additional IDs: town_id, flatm_id (numerical, may be dropped if redundant with town_name, flatm_name).

Preprocessing:





Numerical: Scale floor_area_sqm, remaining_lease_months, year.



Nominal: One-hot encode month, town_name, flatm_name.



Ordinal: Encode flat_type, storey_range with order.



Drop: Remove id, block, street_name, etc., for simplicity.

Pipeline Steps





Data Import: Load resale_transaction.csv using pandas.



EDA and Cleaning: Analyze correlations (e.g., floor_area_sqm 0.71) and remove missing/invalid data in eda.ipynb.



Preprocessing: Scale numerical features, encode categorical features, and convert storey_range to numbers.



Model Development: Train Linear Regression (baseline), Ridge, Lasso, and ElasticNet.



Hyperparameter Tuning: Optimize Ridge and Lasso with GridSearchCV and RandomizedSearchCV.



Evaluation: Compute MAE, MSE, RMSE, and R² on validation and test sets.



Final Model Selection: Choose the best model based on R² and RMSE.



Project Structure

hdb-resale-prediction/
├── data/
│   └── resale_transaction.csv  # HDB dataset
├── outputs/
│   ├── models/                # Trained models (.pkl files)
│   └── plots/                 # Coefficient plots (.png files)
├── src/
│   ├── data_preparation.py    # Data loading and preprocessing
│   ├──model_training.py       # Model training and evaluation
|   └── config.yaml            # Pipeline settings
├── eda.ipynb                  # Exploratory data analysis
├── main.py                    # Main script to run pipeline
├── run.sh                     # Bash script to execute main.py
├── requirements.txt           # Library dependencies
└── README.md                  # This file
