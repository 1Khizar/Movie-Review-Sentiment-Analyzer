import streamlit as st
import pickle
import re
import nltk
import string
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Download required NLTK data
@st.cache_data
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except:
        pass

download_nltk_data()

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# Initialize session state for storing analysis history
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'total_confidence': 0
    }

if 'show_placeholder' not in st.session_state:
    st.session_state['show_placeholder'] = True


# Set page config
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

            
st.markdown("""
<style>

/* Hide Streamlit header, menu & footer */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Remove top padding of the main container */
div.block-container {
    padding-top: 0rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }

/* Main Header */
.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    color: white;
    margin: -1rem -2rem 3rem -2rem;
    border-radius: 0 0 30px 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    position: relative;
}

/* Stat Cards */
.stat-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 25px 20px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    transition: transform 0.3s, box-shadow 0.3s;
    cursor: default;
}
.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 25px rgba(0,0,0,0.3);
}

.stat-number {
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 5px;
}

.stat-label {
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.9;
}

/* Positive and Negative Cards */
.stat-positive { background: linear-gradient(135deg, #28a745, #71e07f); }
.stat-negative { background: linear-gradient(135deg, #dc3545, #ff7b81); }
.stat-confidence { background: linear-gradient(135deg, #f0ad4e, #ffc107); }

/* Result Boxes */
.result-positive, .result-negative {
    background: linear-gradient(135deg, #d4edda, #a1dfb1);
    border: none;
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    color: #155724;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    font-size: 1.2rem;
    width: 100%;
    min-height: 170px; /* Match the text area height (adjust if needed) */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.result-negative { 
    background: linear-gradient(135deg, #f8d7da, #f1a0a8);
    color: #721c24;
}
/* Default Result / Placeholder */
/* Default Result / Placeholder */
.result-placeholder {
    background-color: white;  /* Match textarea background */
    border: 2px solid #667eea; /* Same as textarea border */
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    color: #555555;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1); /* match textarea shadow */
    font-size: 1.2rem;
    width: 100%;
    min-height: 170px; 
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.result-placeholder .placeholder-icon {
    font-size: 2.5rem;
    margin-bottom: 10px;
}
.result-placeholder .placeholder-text {
    font-size: 1.1rem;
    font-weight: 500;
}
.result-placeholder .placeholder-subtext {
    font-size: 0.95rem;
    color: #777777;
    margin-top: 5px;
}


/* Text Area Styling */
textarea {
    border-radius: 15px !important;
    padding: 15px !important;
    border: 2px solid #667eea !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
    font-size: 1rem !important;
}

/* Buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
}
button[kind="secondary"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
}

/* History Items */
.history-item {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    border-left: 5px solid #ddd;
    transition: background-color 0.3s;
}
.history-positive { border-left-color: #28a745; }
.history-negative { border-left-color: #dc3545; }
.history-item:hover { background-color: #e9ecef; }

</style>
""", unsafe_allow_html=True)


# Load model and vectorizer (with error handling)
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('count_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'model.pkl' and 'count_vectorizer.pkl' are in the same directory.")
        return None, None

# Initialize NLTK components
stopwords_set = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Fix the punctuation removal line
    text = ''.join(char for char in text if char not in punctuation)
    
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords_set]
    
    tagged_text = pos_tag(filtered_tokens)
    lemmatized_text = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_text]
    
    return ' '.join(lemmatized_text)

def update_stats(sentiment, confidence):
    st.session_state.stats['total'] += 1
    st.session_state.stats['total_confidence'] += confidence
    
    if sentiment == 'positive':
        st.session_state.stats['positive'] += 1
    else:
        st.session_state.stats['negative'] += 1

def add_to_history(text, sentiment, confidence):
    history_item = {
        'text': text[:100] + ('...' if len(text) > 100 else ''),
        'sentiment': sentiment,
        'confidence': confidence,
        'timestamp': datetime.now()
    }
    st.session_state.analysis_history.insert(0, history_item)
    
    # Keep only last 10 analyses
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history = st.session_state.analysis_history[:10]

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎬 Movie Review Sentiment Analyzer</h1>
    
</div>
""", unsafe_allow_html=True)

# Load models
model, vectorizer = load_model()

if model is not None and vectorizer is not None:
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        def clear_review_text():
            st.session_state["review_text"] = ""
            st.session_state["show_placeholder"] = True  # show placeholder instead of last result

    
        st.markdown("### 📝 Enter Your Movie Review")
        # Text area for review input
        user_input = st.text_area(
            "Movie Review:",
            key = 'review_text',
            height=200,
            placeholder="Type your movie review here...",
            help="Enter a detailed movie review to analyze its sentiment"
        )
        
        # Clear sample text from session state after use
        if 'sample_text' in st.session_state:
            del st.session_state.sample_text
        
        # Character counter
        char_count = len(user_input)
        
        # Analyze button
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        with col_btn2:
            if st.button("🗑️ Clear Text", use_container_width=True, on_click=clear_review_text):
                st.rerun()
                
    
    with col2:
        st.markdown("### 📊 Results")
        st.markdown('<div style="margin-top: 26px;"></div>', unsafe_allow_html=True)

        # Show results if analysis clicked
        if analyze_clicked and user_input.strip():
            with st.spinner("🤖 Analyzing sentiment..."):
                try:
                    # Preprocess and predict
                    processed_text = preprocess(user_input)
                    vect_input = vectorizer.transform([processed_text])
                    prediction = model.predict(vect_input)[0]

                    # Get confidence
                    proba = model.predict_proba(vect_input)[0]
                    confidence = round(max(proba) * 100, 1)

                    st.session_state["show_placeholder"] = False  # hide placeholder

                    # Display result
                    if prediction == 'positive':
                        st.markdown(f"""
                        <div class="result-positive">
                            <div class="sentiment-icon">😊</div>
                            <div class="sentiment-text">Positive</div>
                            <div class="confidence-text">Confidence: {confidence}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-negative">
                            <div class="sentiment-icon">😞</div>
                            <div class="sentiment-text">Negative</div>
                            <div class="confidence-text">Confidence: {confidence}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Update stats and history
                    update_stats(prediction, confidence)
                    add_to_history(user_input, prediction, confidence)
                    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
                    st.success("✅ Analysis completed!")

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

        # Show placeholder if nothing analyzed or after clearing
        
        if st.session_state.get("show_placeholder", True):
            st.markdown(f"""
            <div class="result-placeholder">
                <div class="placeholder-icon">📝</div>
                <div class="placeholder-text">Your result will appear here</div>
                <div class="placeholder-subtext">Type a review and click "Analyze Sentiment"</div>
            </div>
            """, unsafe_allow_html=True)

        
    # Statistics section
    st.markdown("---")
    st.markdown("### 📈 Your Analysis Statistics")
    
    stats = st.session_state.stats
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total']}</div>
            <div class="stat-label">Total Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['positive']}</div>
            <div class="stat-label">Positive Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['negative']}</div>
            <div class="stat-label">Negative Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_confidence = round(stats['total_confidence'] / stats['total'], 1) if stats['total'] > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{avg_confidence}%</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    # Visualization section
    if stats['total'] > 0:
        st.markdown("---")
        st.markdown("### 📊 Sentiment Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for sentiment distribution
            labels = ['Positive', 'Negative']
            values = [stats['positive'], stats['negative']]
            colors = ['#28a745', '#dc3545']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values,
                marker_colors=colors,
                hole=0.4
            )])
            fig_pie.update_layout(
                title="Sentiment Distribution",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence histogram
            if st.session_state.analysis_history:
                confidence_data = [item['confidence'] for item in st.session_state.analysis_history]
                
                fig_hist = go.Figure(data=[go.Histogram(
                    x=confidence_data,
                    nbinsx=10,
                    marker_color='#667eea',
                    opacity=0.7
                )])
                fig_hist.update_layout(
                    title="Confidence Score Distribution",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # History section
if st.session_state.analysis_history:
    st.markdown("---")
    with st.expander("📚 Recent Analysis History", expanded=False):
                
        # Display history entries
        for item in st.session_state.analysis_history[:10]:
            sentiment_class = "history-positive" if item['sentiment'] == 'positive' else "history-negative"
            st.markdown(f"""
            <div style="
                display: flex;
                flex-direction: column;
                padding: 15px 20px;
                margin: 10px 0;
                border-radius: 15px;
                box-shadow: 0 6px 15px rgba(0,0,0,0.1);
                border-left: 6px solid {'#28a745' if item['sentiment']=='positive' else '#dc3545'};
                background-color: #ffffff;
                transition: transform 0.2s;
            " onmouseover="this.style.transform='scale(1.02)';" onmouseout="this.style.transform='scale(1)';">
                <div style="font-size: 15px; color: #333; margin-bottom: 8px; word-break: break-word;">
                    "{item['text']}"
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: bold; font-size: 16px; color: {'#28a745' if item['sentiment']=='positive' else '#dc3545'};">
                        {item['sentiment'].capitalize()} ({item['confidence']}%)
                    </div>
                    <div style="font-size: 12px; color: #888;">
                        {item['timestamp'].strftime('%Y-%m-%d %H:%M')}
                    </div>
                </div>
            </div>
            
            """, unsafe_allow_html=True)
            
                # Delete all button
    if st.button("🗑️ Delete All History"):
            st.session_state.analysis_history = []
            st.session_state.stats = {
                'total': 0,
                'positive': 0,
                'negative': 0,
                'total_confidence': 0
            }
            st.success("✅ History cleared!")
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear All History"):
        st.session_state.analysis_history = []
        st.session_state.stats = {
            'total': 0,
            'positive': 0,
            'negative': 0,
            'total_confidence': 0
        }
        st.success("History cleared!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="
    text-align: center; 
    padding: 15px 0; 
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: #fff;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
">
    <p style="margin: 0; font-size: 1.1rem;">🎬 Movie Review Sentiment Analyzer</p>
    <p style="margin: 0; font-weight: 500; font-size: 0.9rem;">© Made by <strong>Khizar Ishtiaq</strong></p>
</div>
""", unsafe_allow_html=True)

