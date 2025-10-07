import streamlit as st
from transformers import pipeline
import re
import time

NEUTRAL_THRESHOLD = 0.60

MODEL_OPTIONS = {
    "distilgpt2 (fast, small)": "distilgpt2",
    "gpt2 (small)": "gpt2",
    "gpt2-medium (larger, slower)": "gpt2-medium"
}

@st.cache_resource(show_spinner=False)
def load_models(gen_model_name: str):
    sentiment_pipeline = pipeline("sentiment-analysis")
    text_gen_pipeline = pipeline("text-generation", model=gen_model_name)
    return sentiment_pipeline, text_gen_pipeline

def detect_sentiment(sentiment_model, text: str):
    res = sentiment_model(text)[0]
    label = res.get("label", "").upper()
    score = float(res.get("score", 0.0))
    if score < NEUTRAL_THRESHOLD:
        return "NEUTRAL", score
    return label, score

def make_prompt(prefix_template: str, topic: str):
    header = f"{prefix_template}\nWrite one coherent paragraph (3-6 sentences). Stay on topic.\n"
    return f"{header}Topic: {topic}\nParagraph:"

def strip_prefix(full_prompt: str, generated: str):
    out = generated[len(full_prompt):] if generated.startswith(full_prompt) else re.sub(re.escape(full_prompt), "", generated, count=1)
    out = re.sub(r'\s+', ' ', out).strip()
    sentences = re.split(r'(?<=[.!?])\s+', out)
    return " ".join(sentences).strip() if len(sentences) > 1 else out

def choose_best_candidate(sentiment_model, candidates, desired_sentiment):
    best_idx, best_score, best_label = 0, -1.0, None
    evaluations = []
    for i, cand in enumerate(candidates):
        lab, sc = detect_sentiment(sentiment_model, cand)
        evaluations.append((lab, sc))
        score = sc + (0.35 if lab == desired_sentiment else 0.0)
        if desired_sentiment == "NEUTRAL" and sc < NEUTRAL_THRESHOLD:
            score += 0.25
        if score > best_score:
            best_score, best_idx, best_label = score, i, lab
    return best_idx, best_label, best_score, evaluations

st.set_page_config(page_title="AI Text Generator", layout="centered")
st.title("📝 AI Text Generator — Simple Version")

st.markdown("Enter a topic or prompt and generate a sentiment-aligned paragraph.")

prompt = st.text_area("Enter your prompt / topic:", height=120)

col1, col2 = st.columns([1, 1])
with col1:
    model_choice_label = st.selectbox("Model:", list(MODEL_OPTIONS.keys()), index=0)
    gen_model_name = MODEL_OPTIONS[model_choice_label]
with col2:
    manual_sentiment = st.selectbox("Sentiment (optional):", ["Auto Detect", "Positive", "Negative", "Neutral"])

generate_btn = st.button("🚀 Generate Paragraph")

with st.spinner("Loading models…"):
    sentiment_model, text_generator = load_models(gen_model_name)

if generate_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        if manual_sentiment != "Auto Detect":
            desired_sentiment = manual_sentiment.upper()
            st.info(f"Using manual sentiment: **{desired_sentiment}**")
        else:
            detected_label, detected_score = detect_sentiment(sentiment_model, prompt)
            desired_sentiment = detected_label
            st.write(f"Detected sentiment: **{detected_label}** (confidence {detected_score:.2f})")

        prefix_templates = {
            "POSITIVE": "Write a joyful, uplifting paragraph in a warm tone.",
            "NEGATIVE": "Write a somber, reflective paragraph conveying disappointment or sadness.",
            "NEUTRAL": "Write a neutral, factual paragraph without emotional language."
        }
        prefix = prefix_templates.get(desired_sentiment, prefix_templates["NEUTRAL"])
        full_prompt = make_prompt(prefix, prompt)

        st.markdown("**Generating…**")
        t0 = time.time()

        results = text_generator(
            full_prompt,
            max_length=350,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            num_return_sequences=3,
            no_repeat_ngram_size=2,
            repetition_penalty=1.1
        )
        elapsed = time.time() - t0

        raw_candidates = [strip_prefix(full_prompt, r.get("generated_text", "")) for r in results]
        best_idx, best_label, best_score, evaluations = choose_best_candidate(
            sentiment_model, raw_candidates, desired_sentiment
        )
        best_text = raw_candidates[best_idx]

        st.success(f"Done in {elapsed:.1f}s — best candidate (classifier: {best_label})")
        st.subheader("Generated Paragraph:")
        st.write(best_text)

        with st.expander("Show candidate alternatives"):
            for i, cand in enumerate(raw_candidates):
                lab, sc = evaluations[i]
                marker = "✅" if i == best_idx else " "
                st.write(f"Candidate {i+1} {marker} — classifier: {lab} ({sc:.2f})")
                st.write(cand)
                st.markdown("---")

st.markdown("---")
st.markdown("**Sample prompts:**")
st.write("- Positive: `The day I finally achieved my goal after months of hard work`")
st.write("- Negative: `I was betrayed by someone I trusted`")
st.write("- Neutral: `Explain how a solar eclipse occurs`")