"""
Zwigato Delivery Delay Predictor + Natural-Language Manager Assistant
---------------------------------------------------------------------
The original ML prediction pipeline is preserved. The LLM is an additive layer.

UI:
1. Natural-language assistant at the TOP: users can describe an order in rough,
   non-technical language.
2. The same existing ML model runs after all required inputs are available.
3. Descriptive + prescriptive knowledge is generated from the model result.
4. The original manual-parameter prediction form remains BELOW the chat.

Model files expected alongside this script:
    linear_regression_model.joblib
    logistic_regression_model.joblib
"""

import datetime as dt
import json
import time
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Zwigato Delivery Delay Predictor", page_icon="🛵", layout="centered")

LATE_THRESHOLD = 30

# =============================================================================
# EXISTING MODEL CODE — PRESERVED
# =============================================================================
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
    """Original feature-engineering logic, kept unchanged."""
    distance_km = haversine_distance(
        order["restaurant_lat"], order["restaurant_lon"],
        order["delivery_lat"], order["delivery_lon"],
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
    """Same two model calls as the original app."""
    X = build_feature_row(order)
    predicted_minutes = float(linear_model.predict(X)[0])
    late_proba = float(logistic_model.predict_proba(X)[0][1])
    is_late = late_proba >= 0.5
    return X, predicted_minutes, late_proba, is_late


def make_model_context(order, X, predicted_minutes, late_proba, is_late):
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
        "note": "Model associations are not causal proof.",
    }


# =============================================================================
# LLM LAYER
# =============================================================================
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def clean_secret(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().strip("`")
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value or None


# The app supports both providers. If Gemini is configured, it is used first;
# otherwise OpenRouter is used. Gemini 2.5 Flash-Lite currently has a free tier.
OPENROUTER_API_KEY = clean_secret(get_secret("OPENROUTER_API_KEY"))

# Use a specific free model instead of openrouter/free for more predictable
# demo behaviour. The free router selects a model at random and can occasionally
# return an empty completion. If the user has the old value in Secrets, we
# transparently replace it with the stable free model below.
OPENROUTER_MODEL = get_secret("OPENROUTER_MODEL", "qwen/qwen3.8-27b:free")
if OPENROUTER_MODEL.strip() == "openrouter/free":
    OPENROUTER_MODEL = "qwen/qwen3.8-27b:free"

# Free models on OpenRouter rotate often (slugs get renamed, removed or rate-limited),
# so instead of hard-coding a list we ask OpenRouter which models are free *right now*.
# The list is cached for an hour. If that lookup fails, a small static list is used.
_STATIC_FREE_FALLBACKS = [
    "qwen/qwen3.8-27b:free",
    "openrouter/free",   # OpenRouter's own router: picks any currently-free model
]


@st.cache_data(ttl=3600, show_spinner=False)
def get_free_openrouter_models(limit: int = 6) -> List[str]:
    """Return IDs of text models that currently cost $0, largest context first."""
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        r.raise_for_status()
        found = []
        for m in r.json().get("data", []):
            mid = m.get("id", "")
            pricing = m.get("pricing") or {}
            arch = m.get("architecture") or {}
            is_free = str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0"
            text_in = "text" in (arch.get("input_modalities") or ["text"])
            text_out = (arch.get("output_modalities") or ["text"]) == ["text"]
            if mid.endswith(":free") and is_free and text_in and text_out:
                found.append((m.get("context_length") or 0, mid))
        found.sort(reverse=True)
        return [mid for _, mid in found[:limit]]
    except Exception:
        return []


def _extract_text_from_openrouter_response(data: dict) -> str:
    """Read normal Chat Completions text robustly."""
    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    # Some OpenAI-compatible responses may expose content as a list of blocks.
    if isinstance(content, list):
        pieces = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces).strip()

    return ""


def call_openrouter(
    system: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Tuple[Optional[str], Optional[str]]:
    """Call OpenRouter directly without changing the existing ML model."""
    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY is not configured in Streamlit Secrets."

    api_key = clean_secret(OPENROUTER_API_KEY)
    if not api_key:
        return None, "OPENROUTER_API_KEY is empty."

    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "https://streamlit.io/",
        "X-Title": "Zwigato Delivery Delay Predictor",
    }

    # Try the configured model first, then a couple of current free fallbacks.
    models_to_try = []
    for m in [OPENROUTER_MODEL] + get_free_openrouter_models() + _STATIC_FREE_FALLBACKS:
        if m and m not in models_to_try:
            models_to_try.append(m)

    errors = []

    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Prevent thinking-only responses for this application. We need the
            # final answer text / JSON, not a reasoning-only completion.
            "reasoning": {"enabled": False},
        }

        response = None
        for attempt in range(1):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
            except requests.RequestException as exc:
                errors.append(f"{model_name}: request failed: {exc}")
                response = None
                break
            break
        if response is None:
            continue

        if response.status_code == 401:
            return None, (
                "OpenRouter authentication failed (HTTP 401). Check Streamlit Secrets: "
                "OPENROUTER_API_KEY must contain only the key value, without 'Bearer '."
            )

        if not response.ok:
            try:
                body = response.json()
            except ValueError:
                body = response.text[:1000]
            if response.status_code == 402:
                reason = "no credits on the OpenRouter account"
            elif response.status_code == 429:
                reason = "rate-limited"
            elif response.status_code == 404:
                reason = "model not available under this name"
            else:
                reason = str(body)[:120]
            errors.append(f"{model_name}: HTTP {response.status_code} ({reason})")
            continue

        try:
            data = response.json()
        except ValueError as exc:
            errors.append(f"{model_name}: invalid JSON response: {exc}")
            continue

        text = _extract_text_from_openrouter_response(data)

        if text:
            return text, None

        # Empty completion: try the next free endpoint rather than surfacing
        # an opaque error to the user. OpenRouter documents that empty outputs
        # can occur on provider responses.
        errors.append(f"{model_name}: empty completion")

    return None, (
        "All free OpenRouter models are busy right now. Please wait a minute and ask again. "
        + " | ".join(errors[-3:])
    )


def call_llm(
    system: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Tuple[Optional[str], Optional[str]]:
    """Single LLM provider: OpenRouter."""
    return call_openrouter(system, messages, temperature, max_tokens)


# =============================================================================
# NATURAL-LANGUAGE ORDER EXTRACTION
# =============================================================================
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

CHAT_SYSTEM = """
You are the natural-language input layer for the Zwigato delivery-delay prediction app.
The app already has a trained ML model. You NEVER make the prediction yourself.
Your job is only to understand the user's language and extract the input parameters.

Return ONLY valid JSON with this exact structure:
{
  "intent": "new_prediction" | "follow_up" | "general",
  "order": {
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
  },
  "reply": "brief natural-language response"
}

Mapping rules:
- clear/clear sky -> Sunny; stormy/storm -> Stormy; fog/foggy -> Fog;
  windy -> Windy; cloudy/overcast -> Cloudy; sandstorm -> Sandstorms.
- very low/low traffic -> Low; moderate/medium -> Medium; high -> High;
  jammed/gridlock/traffic jam -> Jam.
- bike/motorbike/motorcycle -> motorcycle; e-scooter/electric scooter -> electric_scooter;
  scooter -> scooter; bicycle/cycle -> bicycle.
- snack/quick bite -> Snack; meal/lunch/dinner -> Meal; drink/beverage -> Drinks; buffet -> Buffet.
- poor vehicle -> 0; fair -> 1; good -> 2; excellent/best -> 3.
- Explicitly stated festival yes/no only; never assume.
- Do not invent any numeric value.
- Do not invent coordinates. The ML model uses the restaurant and delivery coordinates directly.
- Do not invent date or exact time.
""".strip()


def blank_order() -> Dict[str, Any]:
    return {key: None for key in FIELD_HELP.keys()}


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def merge_orders(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(old)
    for key in result:
        value = new.get(key) if isinstance(new, dict) else None
        if value is not None and value != "":
            result[key] = value
    return result


def missing_fields(order: Dict[str, Any]) -> List[str]:
    return [key for key, value in order.items() if value is None or value == ""]


def parse_user_order(message: str) -> Tuple[Optional[dict], Optional[str]]:
    draft = st.session_state.get("chat_draft", blank_order())
    prompt = (
        "Current draft from earlier chat messages:\n"
        + json.dumps(draft, indent=2)
        + "\n\nNew user message:\n"
        + message
        + "\n\nMerge the new information into the current draft and return ONLY JSON."
    )
    content, error = call_llm(
        CHAT_SYSTEM,
        [{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1000,
    )
    if error:
        return None, error
    parsed = extract_json(content)
    if not parsed:
        return None, "I could not understand the details clearly. Please describe the order again in simple language."
    return parsed, None


def convert_chat_order(order: Dict[str, Any]) -> dict:
    order_dt = dt.datetime.strptime(str(order["order_datetime"]), "%Y-%m-%d %H:%M")
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


def validate_chat_order(order: dict):
    if not 0 <= order["prep_time"] <= 60:
        raise ValueError("Kitchen preparation time must be between 0 and 60 minutes.")
    if not 0 <= order["multiple_deliveries"] <= 3:
        raise ValueError("Multiple deliveries must be between 0 and 3.")
    if not 15 <= order["age"] <= 50:
        raise ValueError("Rider age must be between 15 and 50.")
    if not 1 <= order["ratings"] <= 6:
        raise ValueError("Rider rating must be between 1 and 6.")
    if not 0 <= order["vehicle_condition"] <= 3:
        raise ValueError("Vehicle condition must be between 0 and 3.")


def generate_manager_insights(context: dict) -> Tuple[Optional[str], Optional[str]]:
    system = """
You are the Zwigato AI Manager Assistant.
The existing ML model has already made the numerical prediction. Treat the supplied
prediction and factor direction as authoritative.

Write two clear sections:

### Descriptive knowledge
Explain what the model predicts, the late-delivery probability, the 30-minute
business threshold, and the main factors that are associated with higher/lower risk.
Do not claim causation.

### Prescriptive knowledge
Provide practical decision-support actions an operations manager could consider.
Tie each action to the supplied prediction/factors. Mention actions such as reviewing
the delivery promise, rider assignment, multiple-delivery load, preparation-time
bottlenecks, routing/traffic exposure, or closer monitoring only where relevant.
Do not invent real-time information, staffing, capacity, costs, or policies.
Do not say an action guarantees prevention of delay.

End with exactly: "Decision support, not autopilot."
""".strip()
    return call_llm(
        system,
        [{"role": "user", "content": "Model context:\n" + json.dumps(context, indent=2)}],
        temperature=0.15,
        max_tokens=1000,
    )


def generate_followup(question: str, context: dict, history: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    system = (
        "You are the Zwigato AI Manager Assistant answering a question about the CURRENT ML prediction.\n"
        "Use the provided model context as the source of truth. Explain the result and provide practical "
        "managerial decision support. Do not change the ML prediction, invent numbers, or claim causation.\n\n"
        "CURRENT MODEL CONTEXT:\n" + json.dumps(context, indent=2)
    )
    messages = history[-8:] + [{"role": "user", "content": question}]
    return call_llm(system, messages, temperature=0.2, max_tokens=800)


# =============================================================================
# SESSION STATE
# =============================================================================
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_draft" not in st.session_state:
    st.session_state.chat_draft = blank_order()
if "last_context" not in st.session_state:
    st.session_state.last_context = None
if "last_ai_insights" not in st.session_state:
    st.session_state.last_ai_insights = None
if "last_ai_error" not in st.session_state:
    st.session_state.last_ai_error = None


# =============================================================================
# TOP NATURAL-LANGUAGE ASSISTANT
# =============================================================================
st.title("🛵 Zwigato Delivery Delay Predictor")
st.caption(
    "Enter the order manually below or describe it here in your own words. "
    "Both options use the same existing ML models."
)

st.subheader("💬 Ask in your own words")
st.caption(
    "Example: *30-year-old rider, rating 4.8, good vehicle, high traffic, rainy weather, "
    "2 other deliveries...*"
)

# Use text_area + button instead of st.chat_input so the chat box is visibly at the TOP.
with st.container(border=True):
    chat_text = st.text_area(
        "Describe the order",
        placeholder=(
            "Example: The rider is 30, has a 4.8 rating, the vehicle is in good condition, "
            "traffic is high and it is raining..."
        ),
        height=100,
        label_visibility="collapsed",
    )
    ask = st.button("Ask Assistant", type="primary", use_container_width=True)

if chat_text and ask:
    st.session_state.chat_messages.append({"role": "user", "content": chat_text})

    with st.chat_message("user"):
        st.markdown(chat_text)

    with st.chat_message("assistant"):
        with st.spinner("Understanding the order details..."):
            parsed, parse_error = parse_user_order(chat_text)

        if parse_error:
            answer = (
                "I couldn't connect to the AI language service right now. "
                "Your manual prediction below is unaffected."
            )
            st.error(parse_error)
        else:
            intent = parsed.get("intent", "new_prediction")
            reply = parsed.get("reply", "")

            if intent == "follow_up" and st.session_state.last_context:
                history = list(st.session_state.chat_messages[:-1])
                answer, follow_error = generate_followup(
                    chat_text, st.session_state.last_context, history
                )
                if follow_error:
                    answer = "I couldn't generate the AI response right now."
                    st.error(follow_error)
                else:
                    st.markdown(answer)

            elif intent == "general" and st.session_state.last_context is None:
                answer = reply or "Describe a delivery order and I will help you assess its delay risk."
                st.markdown(answer)

            else:
                st.session_state.chat_draft = merge_orders(
                    st.session_state.chat_draft,
                    parsed.get("order", {}) or {},
                )
                draft = st.session_state.chat_draft
                missing = missing_fields(draft)

                if missing:
                    friendly = reply or "I have captured some of the order details."
                    human_names = [FIELD_HELP[field] for field in missing]
                    if len(human_names) <= 4:
                        ask_for = ", ".join(human_names)
                    else:
                        ask_for = ", ".join(human_names[:4]) + f", and {len(human_names) - 4} more"
                    answer = f"{friendly}\n\nI still need **{ask_for}** before I can run the existing model."
                    st.markdown(answer)
                else:
                    try:
                        order = convert_chat_order(draft)
                        validate_chat_order(order)
                        X, predicted_minutes, late_proba, is_late = run_existing_prediction(order)
                        context = make_model_context(order, X, predicted_minutes, late_proba, is_late)
                        st.session_state.last_context = context
                        st.session_state.chat_draft = blank_order()

                        st.markdown(
                            f"**Predicted delivery time:** {predicted_minutes:.0f} minutes  \n"
                            f"**Late-delivery probability:** {late_proba * 100:.0f}%  \n"
                            f"**Status:** {'Likely late' if is_late else 'Likely on time'}"
                        )

                        insights, insight_error = generate_manager_insights(context)
                        st.session_state.last_ai_insights = insights
                        st.session_state.last_ai_error = insight_error
                        if insight_error:
                            st.info(
                                "The ML prediction is complete. AI descriptive/prescriptive knowledge "
                                "could not be generated at the moment."
                            )
                            answer = (
                                f"**Predicted delivery time:** {predicted_minutes:.0f} minutes  \n"
                                f"**Late-delivery probability:** {late_proba * 100:.0f}%  \n"
                                f"**Status:** {'Likely late' if is_late else 'Likely on time'}"
                            )
                        else:
                            st.markdown(insights)
                            answer = (
                                f"**Predicted delivery time:** {predicted_minutes:.0f} minutes  \n"
                                f"**Late-delivery probability:** {late_proba * 100:.0f}%  \n"
                                f"**Status:** {'Likely late' if is_late else 'Likely on time'}\n\n"
                                + (insights or "")
                            )
                    except Exception as exc:
                        answer = f"I understood the message, but the existing model could not run: {exc}"
                        st.error(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# Render previous chat in a compact history area.
if st.session_state.chat_messages:
    with st.expander("Conversation history", expanded=False):
        for msg in st.session_state.chat_messages:
            st.markdown(f"**{'You' if msg['role'] == 'user' else 'Assistant'}:** {msg['content']}")
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_messages = []
            st.session_state.chat_draft = blank_order()
            st.rerun()


# =============================================================================
# LAST AI KNOWLEDGE — visible after a prediction
# =============================================================================
if st.session_state.last_context is not None:
    context = st.session_state.last_context
    p = context["prediction"]

    st.divider()
    st.subheader("🤖 AI Manager Knowledge")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted delivery time", f"{p['predicted_delivery_time_minutes']:.0f} min")
    with c2:
        st.metric("Late probability", f"{p['late_probability_percent']:.0f}%")
    with c3:
        st.metric("Model status", p["classification"])

    if st.session_state.last_ai_error:
        st.info("AI explanation is temporarily unavailable; the ML prediction above is still valid.")
    elif st.session_state.last_ai_insights:
        st.markdown(st.session_state.last_ai_insights)

    st.caption("The ML model makes the prediction; the AI layer adds descriptive and prescriptive knowledge.")


# =============================================================================
# EXISTING MANUAL PREDICTION FORM — BELOW THE CHAT
# =============================================================================
st.divider()
st.subheader("🧮 Manual Prediction")
st.caption("Use the original parameter-entry interface when you want complete manual control.")

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

    st.session_state.last_context = make_model_context(
        order, X, predicted_minutes, late_proba, is_late
    )

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

    # Additive LLM layer: descriptive + prescriptive knowledge from the existing ML result.
    if OPENROUTER_API_KEY:
        with st.spinner("Generating descriptive and prescriptive knowledge..."):
            insights, ai_error = generate_manager_insights(st.session_state.last_context)
        st.session_state.last_ai_insights = insights
        st.session_state.last_ai_error = ai_error
        st.divider()
        st.subheader("🤖 AI Manager Knowledge")
        if ai_error:
            st.info("The ML prediction is complete, but the AI explanation could not be generated right now.")
        else:
            st.markdown(insights)
    else:
        st.divider()
        st.subheader("🤖 AI Manager Knowledge")
        st.info(
            "The ML prediction is complete. Add an LLM API key in Streamlit Secrets "
            "to generate descriptive and prescriptive knowledge."
        )

st.divider()
st.caption(
    "Built for the Zwigato delivery-delay case study. Linear regression estimates delivery "
    "minutes; logistic regression estimates the probability of a late delivery, defined as "
    f"delivery time exceeding {LATE_THRESHOLD} minutes — a business rule set for this study, "
    "not a fixed SLA."
)
