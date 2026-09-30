# Zwigato AI Manager — final UI

This version keeps the original prediction model and manual parameter interface,
and adds a natural-language assistant at the top.

## GitHub files required

- app.py
- linear_regression_model.joblib
- logistic_regression_model.joblib
- requirements.txt

## Recommended free LLM setup

Use Gemini in Streamlit Secrets:

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash-lite"
LLM_PROVIDER = "gemini"

The code also supports OpenRouter:

OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_MODEL = "openrouter/free"
LLM_PROVIDER = "openrouter"

In auto mode the app can fall back between the two providers if both keys are configured.

## UI behavior

1. A real text box + Ask Assistant button is shown at the TOP. It is not the Streamlit
   bottom-fixed chat input.
2. Users can describe an order in natural language.
3. The LLM extracts values and asks for missing information; it never predicts.
4. Once complete, the SAME existing feature-engineering and ML models run.
5. Descriptive and prescriptive knowledge is generated from the model result.
6. The original manual prediction interface remains below.
7. The LLM connection/status panel has been removed.

## Important

Do not upload your actual secrets.toml or API keys to GitHub.
Put API keys only in Streamlit Cloud -> Settings -> Secrets.
