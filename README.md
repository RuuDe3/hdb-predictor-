HDB Resale Price Prediction
Class-based ML pipeline for predicting HDB resale prices with baseline and tuned models.
Setup

Clone:git clone https://github.com/your-username/hdb-resale-prediction.git
cd hdb-resale-prediction


Install dependencies:pip install pandas numpy scikit-learn pyyaml joblib (requirements.txt)


Run:./run.sh


To capture logs:python main.py > output.log 2>&1
cat output.log





Outputs

Models: outputs/models/<best_model>.pkl
Metrics: RidgeTuned MAE ~$42,449, R² ~0.86 (validation)
Logs: output.log

Notes

Required columns: floor_area_sqm, storey_range, flat_type, town_name, flatm_name, resale_price, month, remaining_lease_months, lease_commence_date.
Optional: town_id, flatm_id, remaining_lease.
Ensure processing.ordinal.flat_type.order matches flat_type values (after converting FOUR ROOM to 4 ROOM).
Verify dataset columns match data.features and processing sections in config.yaml.
