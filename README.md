# FinSight - Finance News Analyzer (NLP + LLM)

Prototype that tags finance text with sentiment and topic (traditional NLP/ML), then uses an LLM
for summaries, key takeaways and Q&A. Informational only - not financial advice.

## Structure
```
main.py              Streamlit app (Analyze / Dashboard / Business case tabs)
train.py             Trains + evaluates sentiment (Naive Bayes) and topic (Logistic Regression) models
core/preprocess.py   Cleaning + data loading
core/sentiment.py    VADER with finance lexicon + trained ML sentiment
core/topic.py        Topic classifier wrapper
core/llm.py          Ollama / LM Studio / Anthropic API integration
data/sample_articles.csv   Small labeled demo dataset (swap in a larger one)
```

## Setup
```bash
pip install -r requirements.txt
python train.py            # prints accuracy / F1 and saves models/
streamlit run main.py      # opens http://localhost:8501
```

## LLM options (pick in the sidebar)
- **Ollama**: `ollama pull llama3.2` and keep Ollama running (port 11434)
- **LM Studio**: start the local server (port 1234), set the model name
- **Anthropic API**: paste an API key in the sidebar (or set ANTHROPIC_API_KEY)

## Testing
1. Pick a sample article, click Analyze: see VADER score, ML label, topic, summary, takeaways.
2. Ask: "What is the overall outlook?" / "What are the main risks?"
3. Open the Dashboard tab for sentiment by topic and over time.

## Known limitations
- The demo dataset has only 32 rows, so ML sentiment metrics are weak. For real results, train on
  Financial PhraseBank or a Kaggle financial-news dataset (CSV with `text,label,topic,date`).
- News and LLM outputs can be biased or wrong. Not a substitute for professional advice.
