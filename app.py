"""
Zwigato Delivery Delay Predictor + AI Manager Assistant
---------------------------------------------------------
IMPORTANT:
- The existing ML models and feature-engineering/prediction logic are preserved.
- The LLM is an additional layer on top of the existing application.
- The top chat box accepts natural/non-technical language, extracts the inputs,
  and runs the SAME existing models.
- The existing manual parameter form remains below the chat.

Model files expected alongside this script:
    linear_regression_model.joblib
    logistic_regression_model.joblib
"""

import datetime as dt
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import requests

st.set_page_config(
    page_title="Zwigato Delivery Delay Predictor",
    page_icon="🛵",
    layout="centered",
)

LATE_THRESHOLD = 30  # same business rule used in the existing application


# ----------------------------------------------------------------------------
# EXISTING MODEL CODE — PRESERVED
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    linear_model = joblib.load("linear_regression_model.joblib")
    logistic_model = joblib.load("logistic_regression_model.joblib")
    return linear_model, logistic_model


linear_model, logistic_model = load_models()
FEATURE_ORDER = list(linear_model.feature_names_in_)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
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
    """EXISTING feature-engineering logic; intentionally unchanged."""
    distance_km = haversine_distance(
        order["restaurant_lat"],
        order["restaurant_lon"],
        order["delivery_lat"],
        order["delivery_lon"],
    )

    order_dt = order["order_datetime"]
    day_of_week = order_dt.weekday()
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

    one_hot_blocks = {
        "Weatherconditions": (order["weather"], ["Fog", "NaN", "Sandstorms", "Stormy", "Sunny", "Windy"]),
        "Road_traffic_density": (order["traffic"], ["Jam", "Low", "Medium"]),
        "Type_of_order": (order["order_type"], ["Drinks ", "Meal ", "Snack "]),
        "Type_of_vehicle": (order["vehicle_type"], ["motorcycle ", "scooter "]),
        "Festival": (order["festival"], ["Yes"]),
        "City": (order["city"], ["Semi-Urban", "Urban"]),
        "Time_of_Day": (time_of_day, ["Evening", "Morning", "Night"]),
        "Order_DayOfWeek": (day_of_week, [1, 2, 3, 4, 5, 6]),
        "Order_Month": (month, [3, 4]),
        "Order_Hour": (hour, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
    }

    for prefix, (value, categories) in one_hot_blocks.items():
        for cat in categories:
            col = f"{prefix}_{cat}"
            row[col] = 1 if value == cat else 0

    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=FEATURE_ORDER, fill_value=0)
    return df_row


def run_existing_prediction(order: dict):
    """Run the exact same two model calls used by the original app."""
    X = build_feature_row(order)
    predicted_minutes = float(linear_model.predict(X)[0])
    late_proba = float(logistic_model.predict_proba(X)[0][1])
    is_late = late_proba >= 0.5
    return X, predicted_minutes, late_proba, is_late


def make_context(order, X, predicted_minutes, late_proba, is_late):
    coefs = pd.Series(logistic_model.coef_[0], index=FEATURE_ORDER)
    contributions = (coefs * X.iloc[0]).sort_values(key=np.abs, ascending=False)
    top = contributions[contributions != 0].head(6)

    return {
        "prediction": {
            "predicted_delivery_time_minutes": round(predicted_minutes, 2),
            "late_probability_percent": round(late_proba * 100, 2),
            "late_threshold_minutes": LATE_THRESHOLD,
            "classification": "LIKELY LATE" if is_late else "LIKELY ON TIME",
        },
        "order": {
            "order_date": str(order["order_datetime"].date()),
            "order_time": str(order["order_datetime"].strftime("%H:%M")),
            "kitchen_preparation_time_minutes": float(order["prep_time"]),
            "multiple_deliveries": int(order["multiple_deliveries"]),
            "restaurant_latitude": float(order["restaurant_lat"]),
            "restaurant_longitude": float(order["restaurant_lon"]),
            "delivery_latitude": float(order["delivery_lat"]),
            "delivery_longitude": float(order["delivery_lon"]),
            "rider_age": int(order["age"]),
            "rider_rating": float(order["ratings"]),
            "vehicle_condition": int(order["vehicle_condition"]),
            "weather": str(order["weather"]).strip(),
            "traffic": str(order["traffic"]).strip(),
            "vehicle_type": str(order["vehicle_type"]).strip(),
            "order_type": str(order["order_type"]).strip(),
            "festival": str(order["festival"]).strip(),
            "city": str(order["city"]).strip(),
        },
        "model_drivers": [
            {
                "factor": str(name),
                "effect": "Increases late risk" if value > 0 else "Decreases late risk",
                "contribution_score": round(float(value), 6),
            }
            for name, value in top.items()
        ],
        "note": "These are model associations, not causal proof.",
    }


# ----------------------------------------------------------------------------
# LLM CONNECTION — ADDITIVE ONLY
# ----------------------------------------------------------------------------
def get_secret(name, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def clean_secret(value):
    """Normalize a key copied into Streamlit Secrets."""
    if value is None:
        return None
    value = str(value).strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
        value = value[1:-1].strip()
    return value or None


OPENROUTER_API_KEY = clean_secret(get_secret("OPENROUTER_API_KEY"))
OPENROUTER_MODEL = get_secret("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"


def test_openrouter_connection():
    """Check the same bearer key before attempting a chat completion."""
    if not OPENROUTER_API_KEY:
        return False, "OPENROUTER_API_KEY is not configured in Streamlit Secrets."
    try:
        r = requests.get(
            OPENROUTER_KEY_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=20,
        )
        if r.ok:
            return True, "OpenRouter authentication is working."
        return False, f"OpenRouter authentication failed (HTTP {r.status_code}). {r.text[:800]}"
    except requests.RequestException as exc:
        return False, f"Could not reach OpenRouter: {exc}"


def call_llm(messages, temperature=0.2, max_tokens=900):
    if not OPENROUTER_API_KEY:
        return None, "Add OPENROUTER_API_KEY in Streamlit Secrets."

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io/",
        "X-Title": "Zwigato Delivery Delay Predictor",
    }

    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
        if not r.ok:
            try:
                body = r.json()
            except ValueError:
                body = r.text[:1200]
            return None, f"OpenRouter HTTP {r.status_code}: {body}"
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            return None, f"OpenRouter returned no assistant content: {data}"
        return content, None
    except requests.RequestException as exc:
        return None, f"LLM request failed: {exc}"
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return None, f"Could not read OpenRouter response: {exc}"


# ----------------------------------------------------------------------------
# CHAT EXTRACTION
# ----------------------------------------------------------------------------
ALLOWED = {
    "weather": ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy", "NaN"],
    "traffic": ["Low", "Medium", "High", "Jam"],
    "vehicle_type": ["motorcycle ", "scooter ", "electric_scooter ", "bicycle "],
    "order_type": ["Snack ", "Meal ", "Drinks ", "Buffet "],
    "festival": ["No", "Yes"],
    "city": ["Urban", "Metropolitian", "Semi-Urban"],
}

FIELD_HELP = {
    "order_datetime": "order date and exact time",
    "prep_time": "estimated kitchen preparation time in minutes",
    "multiple_deliveries": "number of deliveries on this trip (0-3)",
    "restaurant_lat": "restaurant latitude",
    "restaurant_lon": "restaurant longitude",
    "delivery_lat": "delivery-location latitude",
    "delivery_lon": "delivery-location longitude",
    "age": "rider age",
    "ratings": "rider rating",
    "vehicle_condition": "vehicle condition from 0 to 3",
    "weather": "weather",
    "traffic": "road traffic density",
    "vehicle_type": "vehicle type",
    "order_type": "order type",
    "festival": "whether it is a festival day",
    "city": "city type",
}

CHAT_EXTRACTOR_SYSTEM = f"""
You are the input-understanding layer for a Zwigato delivery-delay prediction app.
The app has an EXISTING machine-learning model. You do not make predictions.
Your only job is to turn a user's natural-language description into the exact
input fields needed by that existing model, or ask for missing fields.

Return ONLY valid JSON. No markdown. Use this exact structure:
{{
  "intent": "new_prediction" | "follow_up" | "general",
  "missing_fields": [],
  "order": {{
    "order_datetime": "YYYY-MM-DD HH:MM" | null,
    "prep_time": number | null,
    "multiple_deliveries": integer | null,
    "restaurant_lat": number | null,
    "restaurant_lon": number | null,
    "delivery_lat": number | null,
    "delivery_lon": number | null,
    "age": number | null,
    "ratings": number | null,
    "vehicle_condition": integer | null,
    "weather": "Sunny" | "Cloudy" | "Fog" | "Sandstorms" | "Stormy" | "Windy" | "NaN" | null,
    "traffic": "Low" | "Medium" | "High" | "Jam" | null,
    "vehicle_type": "motorcycle " | "scooter " | "electric_scooter " | "bicycle " | null,
    "order_type": "Snack " | "Meal " | "Drinks " | "Buffet " | null,
    "festival": "No" | "Yes" | null,
    "city": "Urban" | "Metropolitian" | "Semi-Urban" | null
  }},
  "reply": "A concise natural-language response to the user"
}}

Required fields for a NEW prediction:
{json.dumps(FIELD_HELP, indent=2)}

Natural-language mapping rules:
- clear/clear sky -> Sunny; storm/stormy -> Stormy; foggy -> Fog; windy -> Windy;
  cloudy/overcast -> Cloudy; sandstorm -> Sandstorms.
- very low/low traffic -> Low; moderate/medium -> Medium; high -> High;
  jammed/gridlock/traffic jam -> Jam.
- bike/motorbike -> motorcycle; e-scooter/electric scooter -> electric_scooter;
  scooter -> scooter; bicycle/cycle -> bicycle.
- snack/quick bite -> Snack; meal/lunch/dinner -> Meal; drinks/beverage -> Drinks;
  buffet -> Buffet.
- festival yes/no must be explicit or clearly stated.
- urban/metro/semi-urban map to the listed city categories.
- Vehicle condition language may be mapped only when clear: poor=0, fair=1,
  good=2, excellent/best=3.
- Do NOT invent coordinates. If the user gives a street/city/location name but
  no coordinates, mark the coordinate fields missing. The current ML model uses
  restaurant/delivery coordinates directly and also derives Distance_km from them.
- Do NOT invent date/time. "Evening" without a clock time is missing exact time.
- Do NOT invent ratings, rider age, prep time, or other numeric values.

A follow-up question about an EXISTING prediction should have intent=follow_up.
A general question with no prediction context should have intent=general.
"""


def extract_json_object(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def blank_order():
    return {
        "order_datetime": None,
        "prep_time": None,
        "multiple_deliveries": None,
        "restaurant_lat": None,
        "restaurant_lon": None,
        "delivery_lat": None,
        "delivery_lon": None,
        "age": None,
        "ratings": None,
        "vehicle_condition": None,
        "weather": None,
        "traffic": None,
        "vehicle_type": None,
        "order_type": None,
        "festival": None,
        "city": None,
    }


def merge_orders(old, new):
    merged = dict(old or blank_order())
    for key, value in (new or {}).items():
        if value is not None and value != "":
            merged[key] = value
    return merged


def normalize_order(parsed_order):
    order = merge_orders(blank_order(), parsed_order)
    return order


def missing_order_fields(order):
    missing = []
    for key, value in order.items():
        if value is None or value == "":
            missing.append(key)
    return missing


def convert_to_model_order(order):
    """Convert the chatbot's structured values into the existing app's order dict."""
    try:
        order_dt = dt.datetime.strptime(str(order["order_datetime"]), "%Y-%m-%d %H:%M")
    except Exception as exc:
        raise ValueError("Order date/time must be in YYYY-MM-DD HH:MM format.") from exc

    return {
        "order_datetime": order_dt,
        "prep_time": float(order["prep_time"]),
        "multiple_deliveries": int(order["multiple_deliveries"]),
        "restaurant_lat": float(order["restaurant_lat"]),
        "restaurant_lon": float(order["restaurant_lon"]),
        "delivery_lat": float(order["delivery_lat"]),
        "delivery_lon": float(order["delivery_lon"]),
        "age": int(float(order["age"])),
        "ratings": float(order["ratings"]),
        "vehicle_condition": int(order["vehicle_condition"]),
        "weather": str(order["weather"]),
        "traffic": str(order["traffic"]),
        "vehicle_type": str(order["vehicle_type"]),
        "order_type": str(order["order_type"]),
        "festival": str(order["festival"]),
        "city": str(order["city"]),
    }


def validate_model_order(order):
    checks = [
        (0 <= float(order["prep_time"]) <= 60, "Preparation time must be 0-60 minutes."),
        (0 <= int(order["multiple_deliveries"]) <= 3, "Multiple deliveries must be 0-3."),
        (15 <= int(order["age"]) <= 50, "Rider age must be 15-50."),
        (1 <= float(order["ratings"]) <= 6, "Rider rating must be 1-6."),
        (0 <= int(order["vehicle_condition"]) <= 3, "Vehicle condition must be 0-3."),
        (order["weather"] in ALLOWED["weather"], "Weather value is not supported."),
        (order["traffic"] in ALLOWED["traffic"], "Traffic value is not supported."),
        (order["vehicle_type"] in ALLOWED["vehicle_type"], "Vehicle type is not supported."),
        (order["order_type"] in ALLOWED["order_type"], "Order type is not supported."),
        (order["festival"] in ALLOWED["festival"], "Festival value is not supported."),
        (order["city"] in ALLOWED["city"], "City type is not supported."),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def parse_chat_message(user_message):
    """Use the LLM to extract structured inputs without making the prediction."""
    previous = st.session_state.get("chat_order_draft") or blank_order()
    prompt = (
        "Existing draft from previous messages:\n"
        + json.dumps(previous, indent=2, default=str)
        + "\n\nNew user message:\n"
        + user_message
        + "\n\nMerge the new information into the existing draft. Return ONLY JSON."
    )

    content, error = call_llm(
        [
            {"role": "system", "content": CHAT_EXTRACTOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=900,
    )
    if error:
        return None, error

    parsed = extract_json_object(content)
    if not parsed:
        return None, "The LLM returned an unreadable structured response. Please try again with more specific details."
    return parsed, None


# ----------------------------------------------------------------------------
# DESCRIPTIVE + PRESCRIPTIVE KNOWLEDGE
# ----------------------------------------------------------------------------
INSIGHT_SYSTEM = """
You are the AI Manager layer for Zwigato's EXISTING delivery-delay ML model.
The numerical prediction and factor signs supplied below are authoritative.
Do not recalculate or change them.

Return exactly two sections:
## Descriptive knowledge
Explain what the model predicts, the late probability, threshold, and what the
supplied top factors indicate. Use simple managerial language. Do not claim
causation.

## Prescriptive knowledge
Give practical decision-support actions for an operations manager. Tie each
action to the supplied prediction/factors. Examples can include reviewing the
delivery promise, rider allocation, preparation time, multiple-delivery load,
routing/traffic exposure, and closer monitoring. Do not invent live traffic,
capacity, policy, cost, or staffing information. Do not claim an action is
certain to fix the delay.

End with one sentence: "Decision support, not autopilot."
"""


def generate_insights(context):
    messages = [
        {"role": "system", "content": INSIGHT_SYSTEM},
        {
            "role": "user",
            "content": "Use this exact model context:\n" + json.dumps(context, indent=2, default=str),
        },
    ]
    return call_llm(messages, temperature=0.15, max_tokens=1000)


def generate_follow_up(question, context, history):
    system = f"""
You are the Zwigato Manager Assistant answering a follow-up question about the
CURRENT prediction.

Use the supplied model context as the source of truth. You may explain the
prediction and provide practical managerial decision support. Do not invent
numbers or real-time information. Do not change the ML prediction. Do not claim
causation from association.

Current model context:
{json.dumps(context, indent=2, default=str)}
"""
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": question})
    return call_llm(messages, temperature=0.2, max_tokens=800)


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
for key, default in {
    "chat_messages": [],
    "chat_order_draft": blank_order(),
    "last_prediction_context": None,
    "last_prediction_source": None,
    "last_insights": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ----------------------------------------------------------------------------
# TOP CHATBOT
# ----------------------------------------------------------------------------
st.title("🛵 Zwigato Delivery Delay Predictor")
st.caption(
    "Use the chat for natural-language prediction, or use the full manual form below. "
    "Both paths use the same existing ML models."
)

st.subheader("💬 Ask in your own words")
st.write(
    "Example: *30-year-old rider, rating 4.8, good vehicle, high traffic, rainy weather...* "
    "The assistant will collect anything missing before it runs the model."
)

with st.expander("🔧 LLM connection status", expanded=False):
    if OPENROUTER_API_KEY:
        st.success("OPENROUTER_API_KEY was detected by the app.")
        st.caption(f"Model: `{OPENROUTER_MODEL}`")
        if st.button("Test OpenRouter connection", key="test_openrouter"):
            ok, message = test_openrouter_connection()
            (st.success if ok else st.error)(message)
    else:
        st.error("OPENROUTER_API_KEY is not detected. Add it in Streamlit → Manage app → Settings → Secrets.")

if not OPENROUTER_API_KEY:
    st.info("Add OPENROUTER_API_KEY in Streamlit Secrets to enable the natural-language chatbot. The manual model remains fully usable.")

# Existing chat history appears above the input box.
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_prompt = st.chat_input(
    "Describe the order in normal language... e.g. 'high traffic, rainy, 4.8 rider rating'"
)

if chat_prompt:
    st.session_state.chat_messages.append({"role": "user", "content": chat_prompt})

    with st.chat_message("user"):
        st.markdown(chat_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Understanding your order details..."):
            parsed, error = parse_chat_message(chat_prompt)

        if error:
            answer = error
            st.error(error)
        else:
            intent = parsed.get("intent", "new_prediction")
            parsed_order = normalize_order(parsed.get("order"))

            if intent == "follow_up" and st.session_state.last_prediction_context:
                history = list(st.session_state.chat_messages[:-1])
                answer, follow_error = generate_follow_up(
                    chat_prompt,
                    st.session_state.last_prediction_context,
                    history,
                )
                if follow_error:
                    answer = follow_error
                    st.error(follow_error)
                else:
                    st.markdown(answer)

            elif intent == "general" and not any(v is not None for v in parsed_order.values()):
                answer = parsed.get("reply") or "Please describe an order and I will help turn it into a prediction."
                st.markdown(answer)

            else:
                # Continue the order draft across multiple chat messages.
                st.session_state.chat_order_draft = merge_orders(
                    st.session_state.chat_order_draft,
                    parsed_order,
                )
                draft = st.session_state.chat_order_draft
                missing = missing_order_fields(draft)

                if missing:
                    friendly = parsed.get("reply") or "I have some of the details."
                    names = [FIELD_HELP.get(x, x) for x in missing]
                    # Do not dump 16 technical field names unless necessary.
                    if len(names) <= 4:
                        request_text = ", ".join(names)
                    else:
                        request_text = "; ".join(names[:4]) + f"; and {len(names)-4} more detail(s)"
                    answer = (
                        f"{friendly}\n\n"
                        f"I still need: **{request_text}**. "
                        "Please provide those details in your own words."
                    )
                    st.markdown(answer)
                else:
                    try:
                        order = convert_to_model_order(draft)
                        validate_model_order(order)
                        X, predicted_minutes, late_proba, is_late = run_existing_prediction(order)
                        context = make_context(
                            order, X, predicted_minutes, late_proba, is_late
                        )
                        st.session_state.last_prediction_context = context
                        st.session_state.last_prediction_source = "Chat"
                        st.session_state.chat_order_draft = blank_order()

                        # Generate descriptive + prescriptive knowledge automatically.
                        insight_text, insight_error = generate_insights(context)

                        answer = (
                            f"### Prediction\n"
                            f"**Predicted delivery time:** {predicted_minutes:.0f} minutes  \n"
                            f"**Late-delivery probability:** {late_proba * 100:.0f}%  \n"
                            f"**Status:** {'Likely late' if is_late else 'Likely on time'}\n\n"
                        )

                        if insight_error:
                            answer += (
                                "The ML prediction is complete. I could not generate the AI descriptive/prescriptive explanation yet. "
                                f"\n\n`{insight_error}`"
                            )
                        else:
                            answer += insight_text

                        st.markdown(answer)
                    except Exception as exc:
                        answer = f"I understood the inputs, but could not run the existing model: {exc}"
                        st.error(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


# ----------------------------------------------------------------------------
# LATEST AI OUTPUT ABOVE THE MANUAL MODEL
# ----------------------------------------------------------------------------
if st.session_state.last_prediction_context is not None:
    context = st.session_state.last_prediction_context
    prediction = context["prediction"]

    st.divider()
    st.subheader("📌 Latest AI-assisted result")
    st.caption(
        f"Source: {st.session_state.last_prediction_source or 'Prediction'} · "
        "The ML prediction is unchanged; the AI adds descriptive and prescriptive knowledge."
    )

    a, b, c = st.columns(3)
    with a:
        st.metric("Predicted delivery time", f"{prediction['predicted_delivery_time_minutes']:.0f} min")
    with b:
        st.metric("Late probability", f"{prediction['late_probability_percent']:.0f}%")
    with c:
        st.metric("Model status", prediction["classification"])


# ----------------------------------------------------------------------------
# EXISTING MANUAL MODEL — KEPT BELOW THE CHAT
# ----------------------------------------------------------------------------
st.divider()
st.subheader("🧮 Manual Prediction")
st.caption(
    "Prefer entering every parameter yourself? Use the original model interface below. "
    "This uses the same inputs, feature engineering, and trained models as before."
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
            min_value=0.0,
            max_value=60.0,
            value=15.0,
            step=1.0,
            help="Expected time between the order being placed and the rider picking it up.",
        )
        multiple_deliveries = st.selectbox(
            "Multiple deliveries on this trip", [0, 1, 2, 3], index=1
        )

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
        vehicle_condition = st.selectbox(
            "Vehicle condition (0=poor, 3=best)", [0, 1, 2, 3], index=1
        )

    st.subheader("Conditions")
    col8, col9 = st.columns(2)
    with col8:
        weather = st.selectbox(
            "Weather",
            ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy", "NaN"],
            index=0,
        )
        traffic = st.selectbox(
            "Road traffic density", ["Low", "Medium", "High", "Jam"], index=1
        )
        vehicle_type = st.selectbox(
            "Vehicle type", ["motorcycle ", "scooter ", "electric_scooter ", "bicycle "], index=0
        )
    with col9:
        order_type = st.selectbox(
            "Type of order", ["Snack ", "Meal ", "Drinks ", "Buffet "], index=0
        )
        festival = st.selectbox("Festival day", ["No", "Yes"], index=0)
        city = st.selectbox(
            "City type", ["Urban", "Metropolitian", "Semi-Urban"], index=1
        )

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

    # SAME prediction logic as original application.
    X, predicted_minutes, late_proba, is_late = run_existing_prediction(order)

    # Store latest prediction so the top chat can answer follow-up questions
    # about the manually created prediction.
    st.session_state.last_prediction_context = make_context(
        order, X, predicted_minutes, late_proba, is_late
    )
    st.session_state.last_prediction_source = "Manual form"

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

    st.subheader("What's driving this prediction")
    coefs = pd.Series(logistic_model.coef_[0], index=FEATURE_ORDER)
    contributions = (coefs * X.iloc[0]).sort_values(key=np.abs, ascending=False)
    top_contributions = contributions[contributions != 0].head(6)

    if len(top_contributions) > 0:
        explain_df = pd.DataFrame(
            {
                "Factor": top_contributions.index,
                "Effect on late risk": [
                    "Increases risk" if v > 0 else "Decreases risk"
                    for v in top_contributions
                ],
            }
        )
        st.table(explain_df)
        st.caption(
            "Factors are the active inputs for this order with the largest effect, "
            "positive or negative, on the logistic model's late-delivery score."
        )
    else:
        st.caption("No single input stands out strongly for this particular order.")

    with st.expander("See the exact feature values sent to the models"):
        st.dataframe(X.T.rename(columns={0: "value"}))

    if OPENROUTER_API_KEY:
        with st.spinner("Generating descriptive and prescriptive knowledge..."):
            insight_text, insight_error = generate_insights(
                st.session_state.last_prediction_context
            )
        st.subheader("🤖 AI Manager Insights")
        if insight_error:
            st.warning(insight_error)
        else:
            st.markdown(insight_text)
    else:
        st.info("Add OPENROUTER_API_KEY in Streamlit Secrets to get descriptive and prescriptive AI knowledge for manual predictions.")

st.divider()
st.caption(
    "Built for the Zwigato delivery-delay case study. Linear regression estimates delivery "
    "minutes; logistic regression estimates the probability of a late delivery, defined as "
    f"delivery time exceeding {LATE_THRESHOLD} minutes — a business rule set for this study, "
    "not a fixed SLA."
)
