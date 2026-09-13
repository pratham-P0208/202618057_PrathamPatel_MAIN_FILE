import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# AIRBNB PRICE PREDICTION - TASK 3
# ============================================================

MODEL_PATH = "airbnb_price_prediction_final.pkl"
DATA_PATH = "AB_NYC_2019.csv"

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Airbnb Nightly Price Predictor")
st.write(
    "Enter the details of an Airbnb listing and the trained "
    "Random Forest model will estimate its nightly price."
)

# ------------------------------------------------------------
# Check required files
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' was not found. "
        "Run Task 1 + Task 2 first so that the trained model is created."
    )
    st.stop()

# The CSV is used only to obtain valid category values for the
# dropdowns. The trained model already contains its preprocessing.
if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset '{DATA_PATH}' was not found. "
        "Keep the CSV in the same folder as this app."
    )
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_categories():
    data = pd.read_csv(
        DATA_PATH,
        usecols=[
            "neighbourhood_group",
            "neighbourhood",
            "room_type"
        ]
    )

    neighbourhood_groups = sorted(
        data["neighbourhood_group"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    neighbourhoods = sorted(
        data["neighbourhood"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    room_types = sorted(
        data["room_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return neighbourhood_groups, neighbourhoods, room_types


model = load_model()
neighbourhood_groups, neighbourhoods, room_types = load_categories()

# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Listing Information")

neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    neighbourhood_groups
)

neighbourhood = st.selectbox(
    "Neighbourhood",
    neighbourhoods
)

room_type = st.selectbox(
    "Room Type",
    room_types
)

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input(
        "Latitude",
        min_value=40.0,
        max_value=41.0,
        value=40.7128,
        step=0.0001,
        format="%.4f"
    )

with col2:
    longitude = st.number_input(
        "Longitude",
        min_value=-75.0,
        max_value=-73.0,
        value=-74.0060,
        step=0.0001,
        format="%.4f"
    )

minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=1250,
    value=3,
    step=1
)

number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    max_value=629,
    value=10,
    step=1
)

reviews_per_month = st.number_input(
    "Reviews per Month",
    min_value=0.0,
    max_value=60.0,
    value=1.0,
    step=0.1
)

calculated_host_listings_count = st.number_input(
    "Host Listings Count",
    min_value=1,
    max_value=327,
    value=1,
    step=1
)

availability_365 = st.number_input(
    "Availability per Year (days)",
    min_value=0,
    max_value=365,
    value=180,
    step=1
)

has_review = st.selectbox(
    "Has Previous Review?",
    options=[1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_input_dataframe(
    neighbourhood_group,
    neighbourhood,
    latitude,
    longitude,
    room_type,
    minimum_nights,
    number_of_reviews,
    reviews_per_month,
    calculated_host_listings_count,
    availability_365,
    has_review
):
    # Same reference point used during Task 2
    reference_lat = 40.7128
    reference_lon = -74.0060

    distance_from_center = np.sqrt(
        (latitude - reference_lat) ** 2
        +
        (longitude - reference_lon) ** 2
    )

    availability_ratio = availability_365 / 365.0

    neighbourhood_room = (
        str(neighbourhood_group)
        + "_"
        + str(room_type)
    )

    input_data = pd.DataFrame([{
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "latitude": latitude,
        "longitude": longitude,
        "distance_from_center": distance_from_center,
        "room_type": room_type,
        "neighbourhood_room": neighbourhood_room,
        "log_minimum_nights": np.log1p(max(minimum_nights, 0)),
        "log_number_of_reviews": np.log1p(max(number_of_reviews, 0)),
        "log_reviews_per_month": np.log1p(max(reviews_per_month, 0)),
        "log_calculated_host_listings_count": np.log1p(
            max(calculated_host_listings_count, 0)
        ),
        "availability_ratio": availability_ratio,
        "has_review": has_review
    }])

    return input_data


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Estimate Nightly Price",
    type="primary",
    use_container_width=True
):

    input_df = create_input_dataframe(
        neighbourhood_group=neighbourhood_group,
        neighbourhood=neighbourhood,
        latitude=latitude,
        longitude=longitude,
        room_type=room_type,
        minimum_nights=minimum_nights,
        number_of_reviews=number_of_reviews,
        reviews_per_month=reviews_per_month,
        calculated_host_listings_count=calculated_host_listings_count,
        availability_365=availability_365,
        has_review=has_review
    )

    try:
        prediction = float(
            model.predict(input_df)[0]
        )

        # Prices used for model training were capped at the
        # 99th percentile ($799), so keep the displayed result
        # within a sensible non-negative range.
        prediction = max(0.0, prediction)

        st.success("Prediction generated successfully!")

        st.metric(
            label="Estimated Nightly Price",
            value=f"${prediction:,.2f}"
        )

        st.info(
            "This is an estimated nightly price, not a guaranteed "
            "market price. The model was trained on historical Airbnb "
            "listing data and does not include every factor that can "
            "affect price."
        )

        with st.expander("View processed input features"):
            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as error:
        st.error(
            "Prediction failed. Please check that the model was "
            "created using the Task 1 + Task 2 code provided for this project."
        )
        st.exception(error)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("About the Model")

    st.write(
        "The application uses the final tuned Random Forest "
        "regression pipeline from Task 2."
    )

    st.write("**Main input groups:**")
    st.write(
        "- Location\n"
        "- Room type\n"
        "- Minimum nights\n"
        "- Reviews\n"
        "- Host listing count\n"
        "- Availability"
    )

    st.write("**Evaluation metrics:**")
    st.write(
        "- MAE: approximately $41.79\n"
        "- RMSE: approximately $72.66\n"
        "- Test R²: approximately 0.515"
    )

    st.caption(
        "These values correspond to the current Task 2 run and "
        "may change if the model is retrained."
    )
