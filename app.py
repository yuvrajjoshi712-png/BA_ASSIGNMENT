"""
Zwigato Delivery Delay Predictor
--------------------------------
Streamlit app for Business Analytics Assignment 2 (predictive analytics &
managerial AI adoption). Loads the linear regression (delivery time, minutes)
and logistic regression (probability of being late, >30 min) models trained
in the accompanying Colab notebook, and reproduces that notebook's exact
feature-engineering pipeline so a live order's inputs line up with the
columns each model was trained on.

Model files expected alongside this script:
    linear_regression_model.joblib
    logistic_regression_model.joblib
"""

import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Zwigato Delivery Delay Predictor", page_icon="🛵", layout="centered")

LATE_THRESHOLD = 30  # minutes — same business rule used to train the logistic model


# ----------------------------------------------------------------------------
# Load models
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    linear_model = joblib.load("linear_regression_model.joblib")
    logistic_model = joblib.load("logistic_regression_model.joblib")
    return linear_model, logistic_model


linear_model, logistic_model = load_models()
FEATURE_ORDER = list(linear_model.feature_names_in_)  # identical for both models


# ----------------------------------------------------------------------------
# Feature engineering — mirrors the notebook's clean_data / feature_engineering /
# pd.get_dummies(drop_first=True) pipeline exactly, for a single order.
# ----------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius, km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def get_time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def build_feature_row(order: dict) -> pd.DataFrame:
    """Turn one order's raw inputs into the one-hot-encoded row the models expect."""

    distance_km = haversine_distance(
        order["restaurant_lat"], order["restaurant_lon"],
        order["delivery_lat"], order["delivery_lon"],
    )

    order_dt = order["order_datetime"]
    day_of_week = order_dt.weekday()  # Monday=0 .. Sunday=6, matches Order_Date.dt.dayofweek
    month = order_dt.month
    hour = order_dt.hour
    time_of_day = get_time_of_day(hour)

    row = {
        "Delivery_person_Age": order["age"],
        "Delivery_person_Ratings": order["ratings"],
        "Restaurant_latitude": order["restaurant_lat"],
        "Restaurant_longitude": order["restaurant_lon"],
        "Delivery_location_latitude": order["delivery_lat"],
        "Delivery_location_longitude": order["delivery_lon"],
        "Vehicle_condition": order["vehicle_condition"],
        "multiple_deliveries": order["multiple_deliveries"],
        "Order_Preparation_Time": order["prep_time"],
        "Distance_km": distance_km,
    }

    # One-hot blocks — category names and the dropped (reference) category in each
    # match pd.get_dummies(..., drop_first=True) exactly as run in the notebook.
    # Missing-value tokens ("NaN "/"conditions NaN") became the literal category
    # "NaN" (Weatherconditions) or "None" (Festival, City, Road_traffic_density)
    # in the training data, and those are included as selectable options below
    # so the app can reproduce an order with missing information if needed.

    one_hot_blocks = {
        "Weatherconditions": (order["weather"], ["Fog", "NaN", "Sandstorms", "Stormy", "Sunny", "Windy"]),  # ref: Cloudy
        "Road_traffic_density": (order["traffic"], ["Jam", "Low", "Medium"]),  # ref: High (None never appears in this data)
        "Type_of_order": (order["order_type"], ["Drinks ", "Meal ", "Snack "]),  # ref: Buffet
        "Type_of_vehicle": (order["vehicle_type"], ["motorcycle ", "scooter "]),  # ref: bicycle / electric_scooter
        "Festival": (order["festival"], ["Yes"]),  # ref: No (None never appears in this data)
        "City": (order["city"], ["Semi-Urban", "Urban"]),  # ref: Metropolitian (None never appears in this data)
        "Time_of_Day": (time_of_day, ["Evening", "Morning", "Night"]),  # ref: Afternoon
        "Order_DayOfWeek": (day_of_week, [1, 2, 3, 4, 5, 6]),  # ref: 0 (Monday)
        "Order_Month": (month, [3, 4]),  # ref: any other month (only Mar/Apr seen in training data)
        "Order_Hour": (hour, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),  # ref: 0-7
    }

    for prefix, (value, categories) in one_hot_blocks.items():
        for cat in categories:
            col = f"{prefix}_{cat}"
            row[col] = 1 if value == cat else 0

    df_row = pd.DataFrame([row])
    # Reindex to the exact column order/set the models were trained on;
    # any column not explicitly set above (there shouldn't be any) is filled 0.
    df_row = df_row.reindex(columns=FEATURE_ORDER, fill_value=0)
    return df_row


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🛵 Zwigato Delivery Delay Predictor")
st.caption(
    "Predicts expected delivery time and the probability an order will be late "
    f"(later than {LATE_THRESHOLD} minutes), to help operations managers set "
    "delivery promises and flag orders needing attention."
)

with st.form("order_form"):
    st.subheader("Order details")

    col1, col2 = st.columns(2)
    with col1:
        order_date = st.date_input("Order date", value=dt.date.today())
        order_time = st.time_input("Order time", value=dt.time(19, 0))
    with col2:
        prep_time = st.number_input(
            "Estimated kitchen preparation time (minutes)",
            min_value=0.0, max_value=60.0, value=15.0, step=1.0,
            help="Expected time between the order being placed and the rider picking it up.",
        )
        multiple_deliveries = st.selectbox("Multiple deliveries on this trip", [0, 1, 2, 3], index=1)

    st.subheader("Locations")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Restaurant**")
        restaurant_lat = st.number_input("Restaurant latitude", value=12.9716, format="%.6f")
        restaurant_lon = st.number_input("Restaurant longitude", value=77.5946, format="%.6f")
    with col4:
        st.markdown("**Delivery location**")
        delivery_lat = st.number_input("Delivery latitude", value=13.0500, format="%.6f")
        delivery_lon = st.number_input("Delivery longitude", value=77.6500, format="%.6f")

    st.subheader("Rider")
    col5, col6, col7 = st.columns(3)
    with col5:
        age = st.number_input("Rider age", min_value=15, max_value=50, value=30)
    with col6:
        ratings = st.number_input("Rider rating", min_value=1.0, max_value=6.0, value=4.7, step=0.1)
    with col7:
        vehicle_condition = st.selectbox("Vehicle condition (0=poor, 3=best)", [0, 1, 2, 3], index=1)

    st.subheader("Conditions")
    col8, col9 = st.columns(2)
    with col8:
        weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy", "NaN"], index=0)
        traffic = st.selectbox("Road traffic density", ["Low", "Medium", "High", "Jam"], index=1)
        vehicle_type = st.selectbox("Vehicle type", ["motorcycle ", "scooter ", "electric_scooter ", "bicycle "], index=0)
    with col9:
        order_type = st.selectbox("Type of order", ["Snack ", "Meal ", "Drinks ", "Buffet "], index=0)
        festival = st.selectbox("Festival day", ["No", "Yes"], index=0)
        city = st.selectbox("City type", ["Urban", "Metropolitian", "Semi-Urban"], index=1)

    submitted = st.form_submit_button("Predict delivery outcome")

if submitted:
    order = {
        "order_datetime": dt.datetime.combine(order_date, order_time),
        "prep_time": prep_time,
        "multiple_deliveries": multiple_deliveries,
        "restaurant_lat": restaurant_lat,
        "restaurant_lon": restaurant_lon,
        "delivery_lat": delivery_lat,
        "delivery_lon": delivery_lon,
        "age": age,
        "ratings": ratings,
        "vehicle_condition": vehicle_condition,
        "weather": weather,
        "traffic": traffic,
        "vehicle_type": vehicle_type,
        "order_type": order_type,
        "festival": festival,
        "city": city,
    }

    X = build_feature_row(order)

    predicted_minutes = float(linear_model.predict(X)[0])
    late_proba = float(logistic_model.predict_proba(X)[0][1])
    is_late = late_proba >= 0.5

    st.divider()
    st.subheader("Prediction")

    res1, res2 = st.columns(2)
    with res1:
        st.metric("Predicted delivery time", f"{predicted_minutes:.0f} min")
    with res2:
        st.metric(
            f"Probability of being late (> {LATE_THRESHOLD} min)",
            f"{late_proba * 100:.0f}%",
            delta="Likely late" if is_late else "Likely on time",
            delta_color="inverse" if is_late else "normal",
        )

    if is_late:
        st.warning(
            f"This order is flagged as **likely late** — predicted delivery time "
            f"is {predicted_minutes:.0f} minutes against the {LATE_THRESHOLD}-minute service level."
        )
    else:
        st.success(
            f"This order is flagged as **likely on time** — predicted delivery time "
            f"is {predicted_minutes:.0f} minutes against the {LATE_THRESHOLD}-minute service level."
        )

    # -- Explanation: top drivers, from the logistic model's coefficients --
    st.subheader("What's driving this prediction")
    coefs = pd.Series(logistic_model.coef_[0], index=FEATURE_ORDER)
    contributions = (coefs * X.iloc[0]).sort_values(key=np.abs, ascending=False)
    top_contributions = contributions[contributions != 0].head(6)

    if len(top_contributions) > 0:
        explain_df = pd.DataFrame({
            "Factor": top_contributions.index,
            "Effect on late risk": ["Increases risk" if v > 0 else "Decreases risk" for v in top_contributions],
        })
        st.table(explain_df)
        st.caption(
            "Factors are the active inputs for this order with the largest effect, "
            "positive or negative, on the logistic model's late-delivery score."
        )
    else:
        st.caption("No single input stands out strongly for this particular order.")

    with st.expander("See the exact feature values sent to the models"):
        st.dataframe(X.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Built for the Zwigato delivery-delay case study. Linear regression estimates delivery "
    "minutes; logistic regression estimates the probability of a late delivery, defined as "
    f"delivery time exceeding {LATE_THRESHOLD} minutes — a business rule set for this study, "
    "not a fixed SLA."
)


# =============================================================================
# OPTIONAL LLM LAYER — ADDED WITHOUT CHANGING THE EXISTING PREDICTION PIPELINE
# =============================================================================
# This section only reads the prediction that the existing model already made
# and sends that context to an LLM for:
#   1) Descriptive knowledge: What does the prediction mean? What is driving it?
#   2) Prescriptive knowledge: What can an operations manager do next?
#   3) Follow-up chatbot questions about the current prediction.
#
# The trained models, feature engineering, inputs, prediction calls, and
# existing explanation table above are intentionally left unchanged.
#
# Free option used here: OpenRouter's free router ("openrouter/free").
# Add OPENROUTER_API_KEY to Streamlit Secrets.
# =============================================================================

import json
import os
import urllib.error
import urllib.request


def _get_secret_or_env(name: str, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


OPENROUTER_API_KEY = _get_secret_or_env("OPENROUTER_API_KEY")
OPENROUTER_MODEL = _get_secret_or_env("OPENROUTER_MODEL", "openrouter/free")


def call_openrouter(messages, temperature=0.2):
    """Call OpenRouter without adding another Python package."""
    if not OPENROUTER_API_KEY:
        return None, (
            "LLM is not configured yet. Add OPENROUTER_API_KEY to "
            "Streamlit Secrets to enable the AI assistant."
        )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://streamlit.io/",
            "X-Title": "Zwigato Delivery Delay Predictor",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return content, None
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = str(exc)
        return None, f"OpenRouter HTTP {exc.code}: {body[:1200]}"
    except Exception as exc:
        return None, f"LLM request failed: {exc}"


def _clean_context_value(value):
    """Convert numpy/pandas scalar values into JSON-safe Python values."""
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def build_ai_context(order, predicted_minutes, late_proba, is_late, X, contributions):
    """Create a compact, factual context for the LLM."""
    active = []
    for name, value in contributions.items():
        active.append({
            "factor": str(name),
            "model_effect": "increases late risk" if value > 0 else "decreases late risk",
            "contribution_score": round(float(value), 6),
        })

    order_context = {
        "order_date": str(order["order_datetime"].date()),
        "order_time": str(order["order_datetime"].time()),
        "estimated_kitchen_preparation_time_min": _clean_context_value(order["prep_time"]),
        "multiple_deliveries": _clean_context_value(order["multiple_deliveries"]),
        "restaurant_latitude": _clean_context_value(order["restaurant_lat"]),
        "restaurant_longitude": _clean_context_value(order["restaurant_lon"]),
        "delivery_latitude": _clean_context_value(order["delivery_lat"]),
        "delivery_longitude": _clean_context_value(order["delivery_lon"]),
        "rider_age": _clean_context_value(order["age"]),
        "rider_rating": _clean_context_value(order["ratings"]),
        "vehicle_condition": _clean_context_value(order["vehicle_condition"]),
        "weather": str(order["weather"]).strip(),
        "traffic": str(order["traffic"]).strip(),
        "vehicle_type": str(order["vehicle_type"]).strip(),
        "order_type": str(order["order_type"]).strip(),
        "festival": str(order["festival"]).strip(),
        "city": str(order["city"]).strip(),
    }

    return {
        "prediction": {
            "predicted_delivery_time_minutes": round(float(predicted_minutes), 2),
            "probability_late_percent": round(float(late_proba) * 100, 2),
            "late_threshold_minutes": LATE_THRESHOLD,
            "classification": "LIKELY LATE" if is_late else "LIKELY ON TIME",
        },
        "order": order_context,
        "model_drivers": active,
        "important_note": (
            "The ML model describes statistical association, not causation. "
            "The LLM must not claim that a factor caused the delay unless such "
            "causal evidence is explicitly available."
        ),
    }


DESCRIPTIVE_SYSTEM = """
You are the descriptive analytics layer for a Zwigato delivery-delay model.

The existing ML models are the source of truth. The LLM does not recalculate,
alter, or second-guess the prediction.

Produce DESCRIPTIVE KNOWLEDGE only:
- State the predicted delivery time.
- State the probability that the order is late and the late threshold.
- Explain what the prediction means in plain language.
- Explain the model's strongest active positive/negative factors supplied in
  model_drivers.
- Describe the order context that is relevant.
- Clearly separate model output from interpretation.
- Never claim correlation is causation.
- Never invent missing values or model results.

Use simple language suitable for an operations manager.
"""


PRESCRIPTIVE_SYSTEM = """
You are the prescriptive analytics layer for a Zwigato delivery-delay model.

The existing ML prediction is the source of truth.

Produce PRESCRIPTIVE KNOWLEDGE:
- Recommend practical operational actions that a delivery operations manager
  can take after seeing the prediction.
- Prioritize actions that reduce the operational risk represented by the
  prediction.
- Recommendations can include adjusting the customer delivery promise,
  assigning a nearer/more experienced rider where operationally feasible,
  flagging the order for closer monitoring, checking kitchen preparation,
  managing multiple-delivery load, or reviewing routing/traffic exposure.
- Tie every recommendation to the supplied prediction and model_drivers.
- Do not claim an intervention is guaranteed to reduce delay.
- Do not invent company policies, costs, staffing levels, routes, or real-time
  traffic data.
- Treat recommendations as decision support, not autonomous decisions.

Return:
1. Immediate action
2. Operational action
3. Monitoring action
4. Why these actions fit this prediction

Use simple, managerial language.
"""


def generate_descriptive_knowledge(context):
    messages = [
        {"role": "system", "content": DESCRIPTIVE_SYSTEM},
        {
            "role": "user",
            "content": (
                "Here is the exact context returned by the existing Zwigato "
                "prediction app. Produce the descriptive knowledge.\n\n"
                + json.dumps(context, indent=2, default=str)
            ),
        },
    ]
    return call_openrouter(messages, temperature=0.1)


def generate_prescriptive_knowledge(context):
    messages = [
        {"role": "system", "content": PRESCRIPTIVE_SYSTEM},
        {
            "role": "user",
            "content": (
                "Here is the exact context returned by the existing Zwigato "
                "prediction app. Produce the prescriptive knowledge.\n\n"
                + json.dumps(context, indent=2, default=str)
            ),
        },
    ]
    return call_openrouter(messages, temperature=0.2)


def generate_chat_reply(user_question, context, chat_history):
    chat_system = f"""
You are the Zwigato Manager AI Assistant.

You are answering follow-up questions about ONE prediction made by the
existing ML models.

Never change the model output. Use ONLY the supplied prediction context and
conversation history. You may explain, summarize, compare supplied factors,
and recommend practical managerial actions. Do not invent live information.

Prediction context:
{json.dumps(context, indent=2, default=str)}

Remember:
- descriptive = what the model predicts and what the supplied factors indicate
- prescriptive = what a manager could do next
- association is not causation
- decision support, not autopilot
"""
    messages = [{"role": "system", "content": chat_system}]
    messages.extend(chat_history[-8:])
    messages.append({"role": "user", "content": user_question})
    return call_openrouter(messages, temperature=0.2)


def render_ai_layer(context):
    """
    Additive AI section that reads the last prediction saved by the existing
    prediction flow. It does not alter or re-run the ML models.
    """
    st.divider()
    st.header("🤖 AI Manager Assistant")
    st.caption(
        "The ML model makes the prediction. The LLM adds descriptive and "
        "prescriptive knowledge around that prediction."
    )

    if not OPENROUTER_API_KEY:
        st.info(
            "Add your OpenRouter key to enable the AI layer. "
            "The original prediction will continue to work exactly as before."
        )
        return

    # Keep generated answers visible across Streamlit reruns.
    if "ai_descriptive_answer" not in st.session_state:
        st.session_state.ai_descriptive_answer = None
    if "ai_prescriptive_answer" not in st.session_state:
        st.session_state.ai_prescriptive_answer = None
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    desc_col, pres_col = st.columns(2)

    with desc_col:
        st.subheader("📊 Descriptive knowledge")
        if st.button(
            "Generate descriptive insight",
            key="generate_descriptive",
            use_container_width=True,
        ):
            with st.spinner("Generating descriptive knowledge..."):
                answer, error = generate_descriptive_knowledge(context)
            if error:
                st.error(error)
            else:
                st.session_state.ai_descriptive_answer = answer

        if st.session_state.ai_descriptive_answer:
            st.markdown(st.session_state.ai_descriptive_answer)

    with pres_col:
        st.subheader("🎯 Prescriptive knowledge")
        if st.button(
            "Generate recommended actions",
            key="generate_prescriptive",
            use_container_width=True,
        ):
            with st.spinner("Generating prescriptive knowledge..."):
                answer, error = generate_prescriptive_knowledge(context)
            if error:
                st.error(error)
            else:
                st.session_state.ai_prescriptive_answer = answer

        if st.session_state.ai_prescriptive_answer:
            st.markdown(st.session_state.ai_prescriptive_answer)

    st.subheader("💬 Ask the Manager Assistant")

    for msg in st.session_state.ai_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "Ask: Why is this order risky? What should I do next?",
        key="ai_chat_input",
    )

    if question:
        history_before_question = list(st.session_state.ai_chat_history)

        st.session_state.ai_chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking about the current prediction..."):
                answer, error = generate_chat_reply(
                    question, context, history_before_question
                )
            if error:
                st.error(error)
                answer = f"LLM error: {error}"
            else:
                st.markdown(answer)

        st.session_state.ai_chat_history.append(
            {"role": "assistant", "content": answer}
        )

    if st.button("Clear AI chat", key="clear_ai_chat"):
        st.session_state.ai_chat_history = []
        st.rerun()


# -----------------------------------------------------------------------------
# SAVE THE LAST EXISTING MODEL PREDICTION (ADDITIVE ONLY)
# -----------------------------------------------------------------------------
# This small state block does not change the prediction itself. It only stores
# the already-calculated result so the chatbot can continue working after
# Streamlit reruns caused by chat messages or AI buttons.
if "last_prediction_context" not in st.session_state:
    st.session_state.last_prediction_context = None

if "submitted" in globals() and submitted:
    coefs = pd.Series(logistic_model.coef_[0], index=FEATURE_ORDER)
    contributions_series = (coefs * X.iloc[0]).sort_values(
        key=np.abs, ascending=False
    )
    top_contributions = contributions_series[
        contributions_series != 0
    ].head(6)

    st.session_state.last_prediction_context = build_ai_context(
        order=order,
        predicted_minutes=predicted_minutes,
        late_proba=late_proba,
        is_late=is_late,
        X=X,
        contributions=top_contributions.to_dict(),
    )

    # New prediction = new case, so clear old AI answers and conversation.
    st.session_state.ai_descriptive_answer = None
    st.session_state.ai_prescriptive_answer = None
    st.session_state.ai_chat_history = []

# Render the AI layer whenever a prediction exists, not only on the form-submit
# rerun. This keeps the chatbot usable without changing the existing model UI.
if st.session_state.last_prediction_context is not None:
    render_ai_layer(st.session_state.last_prediction_context)
