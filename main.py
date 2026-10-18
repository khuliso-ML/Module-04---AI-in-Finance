"""FinSight - Finance news analyzer (NLP + LLM). Run: streamlit run main.py"""
import pandas as pd
import streamlit as st
from core import llm
from core.preprocess import load_articles, clean_text
from core.sentiment import vader_sentiment, load_ml_model, ml_sentiment
from core.topic import load_topic_model, predict_topic

st.set_page_config(page_title="FinSight", page_icon="📈", layout="wide")
st.title("📈 FinSight - Finance News Analyzer")
st.caption("Informational prototype only. Not financial advice.")


@st.cache_resource
def get_models():
    return load_ml_model(), load_topic_model()


@st.cache_data
def get_data():
    return load_articles()


sent_model, topic_model = get_models()
df = get_data()

# ---- Sidebar: LLM settings ----
st.sidebar.header("LLM settings")
provider = st.sidebar.selectbox("Provider", ["Ollama (local)", "LM Studio (local)", "Anthropic API"])
default_model = {"Ollama (local)": "llama3.2", "LM Studio (local)": "local-model",
                 "Anthropic API": "claude-sonnet-4-6"}[provider]
model_name = st.sidebar.text_input("Model name", default_model)
api_key = st.sidebar.text_input("Anthropic API key", type="password") if provider == "Anthropic API" else None
cfg = dict(provider=provider, model=model_name, api_key=api_key)
if sent_model is None:
    st.sidebar.warning("No trained models found. Run `python train.py` first.")

tab_analyze, tab_dash, tab_biz = st.tabs(["Analyze", "Dashboard", "Business case"])

# ---- Analyze ----
with tab_analyze:
    mode = st.radio("Input", ["Pick a sample article", "Paste your own"], horizontal=True)
    if mode == "Pick a sample article":
        idx = st.selectbox("Article", df.index, format_func=lambda i: f"[{df.loc[i,'topic']}] {df.loc[i,'text'][:80]}")
        text = df.loc[idx, "text"]
        st.text_area("Article text", text, height=100, disabled=True)
    else:
        text = st.text_area("Paste a finance article or headline", height=160)

    if st.button("Analyze", type="primary") and text.strip():
        v = vader_sentiment(text)
        m = ml_sentiment(sent_model, text)
        topic = predict_topic(topic_model, text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rule-based (VADER)", v["label"].title(), f"score {v['score']}")
        c2.metric("ML (Naive Bayes)", m["label"].title() if m else "n/a",
                  f"conf {m['confidence']}" if m else None)
        c3.metric("Topic", topic or "n/a")
        with st.spinner("Asking the LLM..."):
            st.subheader("Summary")
            st.write(llm.summarize(text, **cfg))
            st.subheader("Key takeaways")
            st.write(llm.insights(text, **cfg))
        st.session_state["current_text"] = text

    st.divider()
    st.subheader("Ask a question about the text")
    q = st.text_input("e.g. What is the overall outlook? What are the main risks?")
    if st.button("Ask") and q.strip() and text.strip():
        with st.spinner("Thinking..."):
            st.write(llm.answer_question(text, q, **cfg))

# ---- Dashboard ----
with tab_dash:
    d = df.copy()
    d["vader_label"] = d["text"].apply(lambda t: vader_sentiment(t)["label"])
    d["vader_score"] = d["text"].apply(lambda t: vader_sentiment(t)["score"])
    agree = (d["vader_label"] == d["label"]).mean()
    st.metric("VADER agreement with labels", f"{agree:.0%}")
    st.subheader("Average sentiment score by topic")
    st.bar_chart(d.groupby("topic")["vader_score"].mean())
    st.subheader("Sentiment trend over time")
    st.line_chart(d.groupby("date")["vader_score"].mean())
    st.subheader("Label distribution")
    st.bar_chart(pd.crosstab(d["topic"], d["label"]))
    st.dataframe(d[["date", "topic", "text", "label", "vader_label", "vader_score"]], use_container_width=True)

# ---- Business case ----
with tab_biz:
    st.markdown("""
**Scenario:** An analyst or news aggregator must triage hundreds of headlines a day.

**Value:** FinSight tags sentiment and sector instantly, summarizes articles, and answers questions
grounded in the text, so analysts spend time on decisions, not reading.

**Use cases:** investment research triage, portfolio-risk early warnings, client newsletter drafting,
compliance monitoring of adverse media.

**Limitations:** Small demo dataset; news and LLM bias; LLM can still err. Always verify.
Not a substitute for professional financial advice.
""")
