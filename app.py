import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for beautification
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    h1 {
        color: #667eea;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 1rem;
        font-size: 0.95rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    .footer a {
        color: white;
        text-decoration: none;
        font-weight: 600;
    }
    .heart {
        color: #ff6b6b;
        animation: heartbeat 1.5s ease-in-out infinite;
    }
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        25% { transform: scale(1.1); }
        50% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🔍 Fake News Detector")
st.markdown('<p class="subtitle">Enter a news article below to check whether it\'s Fake or Real</p>', unsafe_allow_html=True)

# Main input section
col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    news_input = st.text_area(
        "News Article:",
        "",
        height=200,
        placeholder="Paste your news article here..."
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔎 Analyze News"):
        if news_input.strip():
            with st.spinner("Analyzing..."):
                transform_input = vectorizer.transform([news_input])
                prediction = model.predict(transform_input)

                if prediction[0] == 1:
                    st.success("✅ The news is Real!")
                else:
                    st.error("❌ The news is Fake!")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

st.markdown("""
    <div class="footer">
        Made with <span class="heart">❤️</span> by <a href="https://github.com/kaabilcoder" target="_blank">Saurabh Kumar Sahu</a>
    </div>
""", unsafe_allow_html=True)

# Add some spacing at bottom to prevent content from hiding behind footer
st.markdown("<br><br><br>", unsafe_allow_html=True)