import streamlit as st
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="centered"
)

# -------------------- Title --------------------
st.title("🐦 Twitter Sentiment Analysis")
st.write("Analyze tweet sentiment using **NLP + Machine Learning**")

# -------------------- NLTK Setup --------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------- Text Cleaning --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# -------------------- Sentiment + Emoji --------------------
def sentiment_with_emoji(sentiment):
    if sentiment.lower() == "positive":
        return "😄 Positive", "success"
    elif sentiment.lower() == "negative":
        return "😡 Negative", "error"
    elif sentiment.lower() == "neutral":
        return "😐 Neutral", "info"
    else:
        return "🤷 Irrelevant", "warning"

# -------------------- Load Model & Vectorizer --------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, tfidf

model, tfidf = load_model()

# -------------------- Sidebar --------------------
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app predicts **Twitter sentiment** using:
- NLP preprocessing
- TF-IDF Vectorization
- Logistic Regression
""")

st.sidebar.markdown("**Sentiment Classes:**")
st.sidebar.markdown("""
- 😄 Positive  
- 😐 Neutral  
- 😡 Negative  
- 🤷 Irrelevant
""")

# -------------------- User Input --------------------
st.subheader("✍️ Enter a Tweet")

user_input = st.text_area(
    "Type your tweet here:",
    placeholder="I love this game, it's amazing!",
    height=120
)

# -------------------- Prediction --------------------
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        result_text, msg_type = sentiment_with_emoji(prediction)

        if msg_type == "success":
            st.success(f"**Predicted Sentiment:** {result_text}")
        elif msg_type == "error":
            st.error(f"**Predicted Sentiment:** {result_text}")
        elif msg_type == "info":
            st.info(f"**Predicted Sentiment:** {result_text}")
        else:
            st.warning(f"**Predicted Sentiment:** {result_text}")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built with ❤️ using NLP, ML & Streamlit")
