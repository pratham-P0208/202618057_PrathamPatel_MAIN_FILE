# NYC Airbnb Price Prediction — DS605 Lab 4

End-to-end machine learning project that predicts the nightly price of a New York City
Airbnb listing, from raw data to a deployable Streamlit app.

**Dataset:** [Kaggle — New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (`AB_NYC_2019.csv`, ~48.9k listings)

---

## Project structure

```
airbnb_project/
├── data/
│   └── AB_NYC_2019.csv                       # raw dataset
├── notebook/
│   └── Airbnb_Price_Prediction.ipynb         # Task 1 + Task 2 (EDA, cleaning, modeling)
├── app/
│   └── app.py                                # Task 3 (Streamlit app)
├── models/
│   ├── airbnb_price_pipeline.pkl             # saved preprocessing + model pipeline
│   ├── metadata.json                         # feature list, best params, test metrics
│   ├── model_comparison.csv                  # metrics for all candidate models
│   └── feature_importance.csv
├── images/                                   # exported plots (also embedded in the notebook)
├── requirements.txt
└── README.md
```

## How to run

### 1. Set up the environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Re-run the notebook
The trained pipeline is already saved under `models/`, so this step is optional unless you
want to reproduce training from scratch or inspect the analysis.
```bash
jupyter notebook notebook/Airbnb_Price_Prediction.ipynb
```

### 3. Launch the Streamlit app
```bash
cd app
streamlit run app.py
```
Open the URL Streamlit prints (default `http://localhost:8501`), fill in the listing details, and click **Predict nightly price**.

---

## Task 1 — Data Analysis & Preparation (summary)

- **Missing values:** `last_review`/`reviews_per_month` are missing together for listings with zero reviews — this is structural, not random, so `reviews_per_month` was filled with `0` and an explicit `has_reviews` flag was engineered.
- **Outliers:** ~11 listings had `price == 0` (invalid/unbookable) and were dropped. Prices were then clipped to the 1st–99th percentile (**\$30–\$799**) to remove extreme luxury/data-entry outliers. `minimum_nights` values above 365 (clearly erroneous) were also removed.
- **Feature engineering:**
  - `distance_to_center` — Euclidean distance from Times Square, as a proxy for location desirability.
  - `has_reviews` — binary flag distinguishing new/untested listings from established ones.
  - `log_price` (`log1p(price)`) — used as the training target to correct for right-skew; predictions are exponentiated back to dollars for evaluation.
- **Feature selection:** kept 9 numeric + 2 categorical features (`neighbourhood_group`, `room_type`). Dropped `id`, `name`, `host_id`/`host_name` (identifiers/free text) and the raw `neighbourhood` (200+ sparse categories, largely redundant with `neighbourhood_group` + lat/long).
- **Key patterns found:** Manhattan and "Entire home/apt" listings command the highest prices; price rises noticeably closer to central Manhattan/Brooklyn.

## Task 2 — Model Training & Evaluation (summary)

Six regressors were trained and compared (Linear Regression, Ridge, Decision Tree, Random Forest,
Gradient Boosting, XGBoost). **XGBoost** performed best and was tuned with `GridSearchCV` (3-fold CV)
over `n_estimators`, `max_depth`, `learning_rate`, and `subsample`.

| Metric (test set) | Value |
|---|---|
| R² | **0.481** |
| MAE | **$42.92** |
| RMSE | **$75.41** |
| Train R² | 0.580 |

The train/test R² gap (~0.10) is modest, indicating tuning kept overfitting reasonably in check —
plain (untuned) linear models underfit (R² ≈ 0.35), while an untuned decision tree overfits more severely.

**Top predictive features:** `room_type` (entire home vs. private/shared room) dominates by a wide
margin, followed by `distance_to_center` and borough (`neighbourhood_group`).

Full plots (actual vs. predicted, residuals, feature importance) are in the notebook and in `images/`.

## Task 3 — Streamlit Application

`app/app.py` loads the saved pipeline (`models/airbnb_price_pipeline.pkl`) and `metadata.json`,
presents a form for borough, room type, minimum nights, reviews, availability, host listing count,
and latitude/longitude, and returns the estimated nightly price along with an approximate
error range based on the model's test-set MAE.

Tested with realistic inputs, e.g.:
- Manhattan, Entire home/apt, central location → ≈ $300+/night
- Bronx, Private room, few reviews → well under $100/night

To deploy publicly (optional), push this repo to GitHub and deploy via
[Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app/app.py`.

## Task 4 — Final Summary & Limitations

**Main results:** A tuned XGBoost model explains ~48% of the variance in nightly price
(R² = 0.48) with a typical error of ~$43. Location (borough, distance to center) and room type
are the dominant price drivers, consistent with intuition and the exploratory analysis.

**Limitations:**
- **Missing signal:** No access to photos, listing descriptions, amenities, or host response
  rate/ratings — all known to meaningfully affect real Airbnb pricing but absent from this dataset.
- **Static snapshot:** Data is from 2019; it doesn't reflect seasonal demand, current market
  conditions, or post-pandemic pricing shifts.
- **Outlier clipping:** Very cheap or very expensive/luxury listings (outside the \$30–\$799
  range used in training) will be predicted less reliably.
- **Sparse `neighbourhood`:** Using only `neighbourhood_group` (borough) instead of the finer
  `neighbourhood` field trades some precision for a more generalizable, less sparse model.
- **Explained variance ceiling:** An R² around 0.5 is typical for this exact dataset in published
  work — a meaningful share of price variation is inherently unpredictable from these tabular
  features alone.

**Possible future improvements:** incorporate text features from listing names/descriptions
(e.g., TF-IDF or embeddings), use target encoding for the fine-grained `neighbourhood` field,
try stacking/ensembling multiple models, and add external data (e.g., subway proximity, seasonal
demand indices).