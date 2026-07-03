import re
from collections import Counter
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Emotional Sentiment Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


EMOTION_LEXICON = {
    "joy": {
        "happy", "happiness", "joy", "joyful", "glad", "delighted", "excited",
        "great", "good", "amazing", "awesome", "excellent", "love", "loved",
        "wonderful", "pleased", "smile", "smiling", "cheerful", "grateful",
        "proud", "hopeful", "peaceful", "relieved", "fantastic", "positive",
    },
    "sadness": {
        "sad", "sadness", "unhappy", "upset", "cry", "crying", "depressed",
        "lonely", "lost", "hurt", "heartbroken", "miserable", "disappointed",
        "hopeless", "regret", "grief", "pain", "sorrow", "negative", "tired",
    },
    "anger": {
        "angry", "anger", "mad", "furious", "annoyed", "irritated", "hate",
        "hated", "frustrated", "rage", "outraged", "offended", "resent",
        "disgusted", "terrible", "awful", "worst", "bad", "blame",
    },
    "fear": {
        "fear", "afraid", "scared", "anxious", "anxiety", "worried", "worry",
        "nervous", "panic", "panicked", "unsafe", "threat", "risk", "danger",
        "terrified", "uncertain", "doubt", "stress", "stressed",
    },
    "surprise": {
        "surprised", "surprise", "shocked", "unexpected", "unbelievable",
        "sudden", "amazed", "astonished", "wow", "confused", "strange",
        "curious", "wonder", "remarkable",
    },
    "neutral": {
        "okay", "fine", "normal", "regular", "average", "standard", "plain",
        "usual", "typical", "moderate", "stable", "neutral",
    },
}

EMOTION_COLORS = {
    "joy": "#22a06b",
    "love": "#c026d3",
    "sadness": "#3f7fca",
    "anger": "#d64545",
    "fear": "#8b5cf6",
    "surprise": "#d97706",
    "neutral": "#64748b",
}

MODEL_LABELS = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink: #152033;
                --muted: #5b667a;
                --panel: #ffffff;
                --line: #d9e1ec;
                --accent: #0f766e;
                --accent-dark: #115e59;
                --soft: #f4f7fb;
            }

            .stApp {
                background:
                    radial-gradient(circle at 12% 18%, rgba(15, 118, 110, 0.08), transparent 25%),
                    linear-gradient(135deg, #f7fafc 0%, #eef3f8 52%, #f8fafc 100%);
                color: var(--ink);
            }

            [data-testid="stSidebar"] {
                background: #102033;
            }

            [data-testid="stSidebar"] * {
                color: #f8fafc;
            }

            .main .block-container {
                max-width: 1180px;
                padding-top: 2.2rem;
                padding-bottom: 2.5rem;
            }

            .hero {
                padding: 2.2rem 2.3rem;
                border: 1px solid rgba(15, 118, 110, 0.18);
                border-radius: 8px;
                background:
                    linear-gradient(120deg, rgba(255, 255, 255, 0.96), rgba(244, 248, 252, 0.92)),
                    linear-gradient(90deg, rgba(15, 118, 110, 0.1), rgba(63, 127, 202, 0.08));
                box-shadow: 0 18px 45px rgba(21, 32, 51, 0.08);
                margin-bottom: 1.25rem;
            }

            .hero h1 {
                font-size: clamp(2rem, 4vw, 3.35rem);
                line-height: 1.04;
                margin: 0 0 0.75rem 0;
                letter-spacing: 0;
                color: #102033;
            }

            .hero p {
                font-size: 1.05rem;
                line-height: 1.7;
                color: var(--muted);
                max-width: 780px;
                margin: 0;
            }

            .section-title {
                font-size: 1.05rem;
                font-weight: 750;
                color: #152033;
                margin: 0.25rem 0 0.75rem 0;
            }

            .metric-panel {
                border: 1px solid var(--line);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.92);
                padding: 1rem 1.05rem;
                min-height: 112px;
                box-shadow: 0 12px 30px rgba(21, 32, 51, 0.06);
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.78rem;
                text-transform: uppercase;
                font-weight: 700;
                letter-spacing: 0.08em;
                margin-bottom: 0.45rem;
            }

            .metric-value {
                color: #102033;
                font-size: 1.7rem;
                line-height: 1.15;
                font-weight: 800;
            }

            .emotion-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 120px;
                padding: 0.58rem 0.8rem;
                border-radius: 8px;
                color: #ffffff;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .insight-box {
                border-left: 4px solid var(--accent);
                background: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                padding: 1rem 1.15rem;
                color: var(--muted);
                line-height: 1.65;
                box-shadow: 0 10px 25px rgba(21, 32, 51, 0.05);
            }

            .stTextArea textarea {
                border-radius: 8px;
                border: 1px solid var(--line);
                font-size: 1rem;
                line-height: 1.55;
            }

            div.stButton > button,
            div.stDownloadButton > button {
                border-radius: 8px;
                border: 1px solid var(--accent);
                background: var(--accent);
                color: #ffffff;
                font-weight: 750;
                min-height: 2.75rem;
            }

            div.stButton > button:hover,
            div.stDownloadButton > button:hover {
                border-color: var(--accent-dark);
                background: var(--accent-dark);
                color: #ffffff;
            }

            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1rem;
            }

            hr {
                border: none;
                border-top: 1px solid var(--line);
                margin: 1.5rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_trained_model():
    app_dir = Path(__file__).resolve().parent
    model_path = app_dir / "model.joblib"
    vectorizer_path = app_dir / "vectorizer.joblib"

    if not model_path.exists() or not vectorizer_path.exists():
        return None, None

    return joblib.load(model_path), joblib.load(vectorizer_path)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def analyze_with_lexicon(text: str) -> dict:
    tokens = tokenize(text)
    token_counts = Counter(tokens)
    scores = {}

    for emotion, words in EMOTION_LEXICON.items():
        raw_score = sum(token_counts[word] for word in words)
        scores[emotion] = raw_score

    total_hits = sum(scores.values())
    if not text.strip():
        dominant = "neutral"
        confidence = 0.0
    elif total_hits == 0:
        dominant = "neutral"
        confidence = 0.42
        scores["neutral"] = max(scores["neutral"], 1)
        total_hits = sum(scores.values())
    else:
        dominant = max(scores, key=scores.get)
        confidence = scores[dominant] / total_hits

    polarity = (
        scores["joy"] * 1.0
        + scores["surprise"] * 0.35
        - scores["sadness"] * 0.85
        - scores["anger"] * 1.0
        - scores["fear"] * 0.7
    )
    normalized_polarity = max(-1.0, min(1.0, polarity / max(total_hits, 1)))

    score_rows = [
        {
            "emotion": emotion.title(),
            "score": score,
            "share": score / max(total_hits, 1),
            "color": EMOTION_COLORS[emotion],
        }
        for emotion, score in scores.items()
    ]

    return {
        "text": text,
        "dominant_emotion": dominant,
        "confidence": round(confidence, 3),
        "polarity": round(normalized_polarity, 3),
        "word_count": len(tokens),
        "scores": scores,
        "score_table": pd.DataFrame(score_rows),
        "source": "Lexicon fallback",
    }


def analyze_text(text: str, model=None, vectorizer=None) -> dict:
    fallback = analyze_with_lexicon(text)
    if model is None or vectorizer is None or not text.strip():
        return fallback

    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    dominant = MODEL_LABELS.get(int(prediction), str(prediction))

    model_emotions = ["sadness", "anger", "love", "surprise", "fear", "joy"]
    scores = {emotion: 0.0 for emotion in model_emotions}
    confidence = fallback["confidence"]

    if hasattr(model, "decision_function"):
        decision = model.decision_function(vectorized_text)
        values = decision[0] if getattr(decision, "ndim", 1) > 1 else decision
        shifted = values - values.min()
        denominator = shifted.sum()
        if denominator > 0:
            probabilities = shifted / denominator
            for label, probability in zip(model.classes_, probabilities):
                emotion = MODEL_LABELS.get(int(label), str(label))
                scores[emotion] = float(probability)
            confidence = float(max(probabilities))

    if not any(scores.values()):
        scores[dominant] = 1.0

    positive = scores.get("joy", 0) + scores.get("love", 0) + (scores.get("surprise", 0) * 0.25)
    negative = scores.get("sadness", 0) + scores.get("anger", 0) + (scores.get("fear", 0) * 0.8)
    polarity = max(-1.0, min(1.0, positive - negative))

    score_rows = [
        {
            "emotion": emotion.title(),
            "score": round(score, 4),
            "share": score,
            "color": EMOTION_COLORS[emotion],
        }
        for emotion, score in scores.items()
    ]

    return {
        "text": text,
        "dominant_emotion": dominant,
        "confidence": round(confidence, 3),
        "polarity": round(polarity, 3),
        "word_count": fallback["word_count"],
        "scores": scores,
        "score_table": pd.DataFrame(score_rows),
        "source": "Trained LinearSVC model",
    }


def score_chart(score_table: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(score_table)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("score:Q", title="Emotion score"),
            y=alt.Y("emotion:N", title=None, sort="-x"),
            color=alt.Color("emotion:N", scale=alt.Scale(
                domain=score_table["emotion"].tolist(),
                range=score_table["color"].tolist(),
            ), legend=None),
            tooltip=[
                alt.Tooltip("emotion:N", title="Emotion"),
                alt.Tooltip("score:Q", title="Score"),
                alt.Tooltip("share:Q", title="Share", format=".0%"),
            ],
        )
        .properties(height=260)
    )


def distribution_chart(results: pd.DataFrame) -> alt.Chart:
    counts = (
        results["dominant_emotion"]
        .str.title()
        .value_counts()
        .rename_axis("emotion")
        .reset_index(name="count")
    )
    counts["color"] = counts["emotion"].str.lower().map(EMOTION_COLORS)

    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("emotion:N", title=None, sort="-y"),
            y=alt.Y("count:Q", title="Records"),
            color=alt.Color("emotion:N", scale=alt.Scale(
                domain=counts["emotion"].tolist(),
                range=counts["color"].tolist(),
            ), legend=None),
            tooltip=["emotion:N", "count:Q"],
        )
        .properties(height=320)
    )


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result: dict) -> None:
    dominant = result["dominant_emotion"]
    color = EMOTION_COLORS.get(dominant, EMOTION_COLORS["neutral"])

    top_left, top_mid, top_right = st.columns([1.1, 1, 1])
    with top_left:
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-label">Detected emotion</div>
                <div class="emotion-badge" style="background:{color};">{dominant}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_mid:
        render_metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    with top_right:
        render_metric("Polarity", f"{result['polarity']:+.2f}")

    st.altair_chart(score_chart(result["score_table"]), use_container_width=True)

    if result["polarity"] > 0.25:
        tone = "The text leans positive, with language that indicates satisfaction, optimism, or enthusiasm."
    elif result["polarity"] < -0.25:
        tone = "The text leans negative, with signals of concern, frustration, sadness, or emotional pressure."
    else:
        tone = "The text is emotionally balanced or neutral, with no strong directional sentiment."

    st.markdown(f'<div class="insight-box">{tone}</div>', unsafe_allow_html=True)


def analyze_dataframe(frame: pd.DataFrame, text_column: str, model=None, vectorizer=None) -> pd.DataFrame:
    rows = []
    for value in frame[text_column].fillna("").astype(str):
        result = analyze_text(value, model, vectorizer)
        rows.append(
            {
                "text": value,
                "dominant_emotion": result["dominant_emotion"],
                "confidence": result["confidence"],
                "polarity": result["polarity"],
                "word_count": result["word_count"],
            }
        )
    return pd.DataFrame(rows)


inject_styles()
model, vectorizer = load_trained_model()
model_status = "Trained model loaded" if model is not None and vectorizer is not None else "Demo analyzer active"

st.sidebar.title("Project Controls")
st.sidebar.caption("Select how you want to test the emotion classifier.")
analysis_mode = st.sidebar.radio(
    "Input type",
    ["Single text", "CSV batch"],
)
show_details = st.sidebar.toggle("Show detailed results", value=True)
st.sidebar.divider()
st.sidebar.markdown("**Model status**")
st.sidebar.caption(model_status)
st.sidebar.caption("The app uses model.joblib and vectorizer.joblib from this project folder.")
with st.sidebar.expander("How to read results"):
    st.caption("Confidence shows how strongly the model supports the predicted emotion.")
    st.caption("Polarity ranges from -1 to +1. Negative values show negative tone, positive values show positive tone.")

st.markdown(
    """
    <section class="hero">
        <h1>Emotional Sentiment Analysis</h1>
        <p>
            A professional data science interface for detecting emotion, confidence,
            and sentiment polarity from typed text or uploaded CSV datasets.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

if analysis_mode == "Single text":
    st.markdown('<div class="section-title">Text Input</div>', unsafe_allow_html=True)
    sample = (
        "I am really happy with the progress, but I still feel a little nervous "
        "about the final presentation."
    )
    text = st.text_area(
        "Enter text for emotional sentiment analysis",
        value=sample,
        height=170,
        label_visibility="collapsed",
    )

    run = st.button("Run Prediction", use_container_width=True)
    if run or text.strip():
        st.divider()
        st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
        result = analyze_text(text, model, vectorizer)
        render_result(result)
        st.caption(f"Prediction source: {result['source']}")

        if show_details:
            st.markdown('<div class="section-title">Emotion Score Table</div>', unsafe_allow_html=True)
            st.dataframe(
                result["score_table"][["emotion", "score", "share"]],
                use_container_width=True,
                hide_index=True,
            )

else:
    st.markdown('<div class="section-title">Dataset Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file, then select the column that contains the sentences or messages.")
    else:
        data = pd.read_csv(uploaded_file)
        text_columns = list(data.columns)

        if not text_columns:
            st.error("No columns were found in this CSV file.")
        else:
            default_index = text_columns.index("text") if "text" in text_columns else 0
            selected_column = st.selectbox(
                "Text column for prediction",
                text_columns,
                index=default_index,
            )
            results = analyze_dataframe(data, selected_column, model, vectorizer)

            st.divider()
            metric_a, metric_b, metric_c, metric_d = st.columns(4)
            with metric_a:
                render_metric("Rows analyzed", f"{len(results):,}")
            with metric_b:
                dominant = results["dominant_emotion"].mode().iloc[0].title()
                render_metric("Most common", dominant)
            with metric_c:
                render_metric("Avg confidence", f"{results['confidence'].mean() * 100:.1f}%")
            with metric_d:
                render_metric("Avg polarity", f"{results['polarity'].mean():+.2f}")

            st.markdown('<div class="section-title">Emotion Distribution</div>', unsafe_allow_html=True)
            st.altair_chart(distribution_chart(results), use_container_width=True)

            if show_details:
                st.markdown('<div class="section-title">Analyzed Records</div>', unsafe_allow_html=True)
                st.dataframe(results, use_container_width=True, hide_index=True)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                data=csv,
                file_name="emotion_analysis_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
