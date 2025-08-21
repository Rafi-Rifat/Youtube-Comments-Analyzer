import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import pickle
import joblib
import requests
import json
from datetime import datetime
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    st.warning("TensorFlow not available. Deep learning models will not work.")

# YouTube Data API configuration
YOUTUBE_API_KEY = 'AIzaSyBJTxJhfAK3e8N8pqPrSg3VCmv4crijJPw'
YOUTUBE_API_BASE_URL = 'https://www.googleapis.com/youtube/v3'

# Set page config at the very top
st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern, professional CSS styling (adjusted for compactness)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, #root, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        margin: 0 !important;
        padding: 0 !important;
        height: 100%;
    }
    
    /* HEADER - ABSOLUTE TOP */
    .header-container {
        position: relative;
        padding: 0.8rem 1rem;
        background: white;
        border-bottom: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-top: -1.5rem !important;
        top: -8px !important;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a202c;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 6px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 0.5rem 0.6rem !important;
        font-size: 0.8rem !important;
        height: 40px !important;
        min-width: 100%;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1a73e8 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        height: 40px !important;
        min-width: 120px !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background-color: #1765cc !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    
    .card-header {
        font-size: 1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    /* Video info */
    .video-info {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }
    
    .video-thumbnail {
        flex-shrink: 0;
        width: 140px;
        height: 78px;
        border-radius: 6px;
        overflow: hidden;
    }
    
    .video-thumbnail img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .video-details {
        flex: 1;
    }
    
    .video-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 0.4rem;
        line-height: 1.3;
    }
    
    .video-meta {
        font-size: 0.8rem;
        color: #4a5568;
        margin-bottom: 0.2rem;
    }
    
    .video-stats {
        display: flex;
        gap: 0.8rem;
        font-size: 0.8rem;
        color: #4a5568;
    }
    
    /* Sentiment grid */
    .sentiment-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }
    
    .sentiment-item {
        background: #f8fafc;
        border-radius: 6px;
        padding: 0.8rem;
        text-align: center;
        border-left: 3px solid #e2e8f0;
    }
    
    .sentiment-item.positive { border-left-color: #38a169; }
    .sentiment-item.negative { border-left-color: #e53e3e; }
    .sentiment-item.neutral { border-left-color: #d69e2e; }
    
    .sentiment-value {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0.4rem 0;
        color: #4a5568;
    }
    
    .sentiment-label {
        font-size: 0.8rem;
        color: #4a5568;
    }
    
    /* Comments */
    .comments-container {
        padding-right: 0.5rem; 
    }
    
    .comment-section-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        padding: 0.4rem;
        border-radius: 4px;
        margin-bottom: 0.6rem;
        text-align: center;
    }
    
    .comment-section-header.positive {
        background: #38a169;
    }
    
    .comment-section-header.negative {
        background: #e53e3e;
    }
    .comment-section-header.neutral {
        background: #d69e2e;
    }
    
    .comment-item {
        background: #f8fafc;
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #e2e8f0;
    }
    
    .comment-text {
        font-size: 0.85rem;
        line-height: 1.4;
        color: #2d3748;
        margin-bottom: 0.4rem;
    }
    
    .comment-meta {
        font-size: 0.75rem;
        color: #718096;
        display: flex;
        justify-content: space-between;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        color: #718096;
        font-size: 0.85rem;
        padding: 1.5rem 0.8rem;
        background: #f8fafc;
        border-radius: 6px;
        border: 1px dashed #cbd5e0;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'comments_data' not in st.session_state:
    st.session_state.comments_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'video_info' not in st.session_state:
    st.session_state.video_info = None

MAX_COMMENTS = 1000  # Maximum number of comments to fetch

@st.cache_resource
def load_models():
    """Load trained models and preprocessing components"""
    try:
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        ensemble_package = joblib.load('models/smart_gating_ensemble.pkl')
        
        required_components = [
            'svm_model', 'bilstm_model', 'gating_classifier',
            'tokenizer', 'max_sequence_length'
        ]
        
        if not all(key in ensemble_package for key in required_components):
            st.error("Missing components in ensemble package")
            return None, None, None, None, None, None
            
        return (
            ensemble_package,
            tfidf_vectorizer,
            label_encoder,
            ensemble_package['tokenizer'],
            {'best_model_type': 'smart_gating'},
            ensemble_package['max_sequence_length']
        )
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None, None

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info(video_id):
    """Fetch video information from YouTube Data API"""
    try:
        url = f"{YOUTUBE_API_BASE_URL}/videos"
        params = {
            'part': 'snippet,statistics',
            'id': video_id,
            'key': YOUTUBE_API_KEY
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data['items']:
            return None

        video = data['items'][0]
        snippet = video['snippet']
        statistics = video['statistics']

        def format_number(num_str):
            try:
                num = int(num_str)
                if num >= 1000000:
                    return f"{num/1000000:.1f}M"
                elif num >= 1000:
                    return f"{num/1000:.1f}K"
                return str(num)
            except:
                return num_str

        return {
            'title': snippet['title'],
            'description': snippet['description'][:150] + "..." if len(snippet['description']) > 150 else snippet['description'],
            'channel_title': snippet['channelTitle'],
            'published_at': snippet['publishedAt'][:10],
            'thumbnail_url': snippet['thumbnails']['high']['url'],
            'view_count': format_number(statistics.get('viewCount', '0')),
            'like_count': format_number(statistics.get('likeCount', '0')),
            'comment_count': format_number(statistics.get('commentCount', '0')),
        }
    except Exception as e:
        st.error(f"Error fetching video info: {str(e)}")
        return None

def fetch_youtube_comments(video_id):
    """Fetch YouTube comments using YouTube Data API"""
    try:
        comments_data = []
        next_page_token = None
        total_fetched = 0
        
        progress_bar = st.progress(0, text="Fetching comments...")
        
        while True:
            url = f"{YOUTUBE_API_BASE_URL}/commentThreads"
            params = {
                'part': 'snippet',
                'videoId': video_id,
                'key': YOUTUBE_API_KEY,
                'maxResults': 100,
                'order': 'relevance'
            }

            if next_page_token:
                params['pageToken'] = next_page_token

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get('items'):
                break

            for item in data['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments_data.append({
                    'comment': comment['textDisplay'],
                    'author': comment['authorDisplayName'],
                    'published_at': comment['publishedAt'][:10],
                    'like_count': comment['likeCount']
                })
                total_fetched += 1
                
                progress_bar.progress(min(total_fetched/MAX_COMMENTS, 1.0), 
                                    text=f"Fetched {total_fetched} comments")
                
                if total_fetched >= MAX_COMMENTS:
                    break

            if total_fetched >= MAX_COMMENTS or not data.get('nextPageToken'):
                break
                
            next_page_token = data.get('nextPageToken')
            time.sleep(0.1)

        progress_bar.empty()
        return comments_data
    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")
        return []

def predict_sentiment(comments, model, tfidf_vectorizer, label_encoder, tokenizer, model_info, max_length):
    """Predict sentiment using Smart Gating Ensemble"""
    processed_comments = [preprocess_text(comment) for comment in comments]
    sentiments = []
    confidence_scores = []
    gating_decisions = []

    if model_info.get('best_model_type') != 'smart_gating':
        st.error("Incorrect model type loaded")
        return ["Error"] * len(comments), [0.0] * len(comments)

    svm_model = model['svm_model']
    bilstm_model = model['bilstm_model']
    gate_clf = model['gating_classifier']
    ensemble_tokenizer = model['tokenizer']
    ensemble_max_length = model['max_sequence_length']

    svm_features = tfidf_vectorizer.transform(processed_comments)
    svm_probs = svm_model.predict_proba(svm_features)
    svm_preds = np.argmax(svm_probs, axis=1)

    if not DEEP_LEARNING_AVAILABLE:
        st.error("TensorFlow required for BiLSTM predictions")
        return ["Error"] * len(comments), [0.0] * len(comments)
        
    sequences = ensemble_tokenizer.texts_to_sequences(processed_comments)
    padded_sequences = pad_sequences(sequences, maxlen=ensemble_max_length)
    bilstm_probs = bilstm_model.predict(padded_sequences)
    bilstm_preds = np.argmax(bilstm_probs, axis=1)

    meta_features = np.hstack([svm_probs, bilstm_probs])

    for i in range(len(processed_comments)):
        default_pred = bilstm_preds[i]
        
        if svm_preds[i] != bilstm_preds[i]:
            gate_pred = gate_clf.predict(meta_features[i:i+1])[0]
            
            if gate_pred == svm_preds[i]:
                final_pred = svm_preds[i]
                gating_decisions.append(True)
            else:
                final_pred = default_pred
                gating_decisions.append(False)
        else:
            final_pred = default_pred
            gating_decisions.append(False)
        
        if final_pred == svm_preds[i]:
            confidence = svm_probs[i][final_pred]
        else:
            confidence = bilstm_probs[i][final_pred]
            
        sentiments.append(label_encoder.inverse_transform([final_pred])[0])
        confidence_scores.append(confidence)

    st.session_state.gating_stats = {
        'total': len(gating_decisions),
        'svm_overrides': sum(gating_decisions),
        'override_pct': (sum(gating_decisions) / len(gating_decisions)) * 100
    }
    
    return sentiments, confidence_scores

def calculate_weighted_percentages(results_df):
    """Calculate sentiment percentages weighted by comment likes"""
    if results_df.empty:
        return 0, 0, 0, 0
    
    # Calculate weighted sums
    positive_weighted = sum(1 + row['like_count'] 
                          for _, row in results_df.iterrows() 
                          if row['sentiment'] == 'Positive')
    
    negative_weighted = sum(1 + row['like_count'] 
                          for _, row in results_df.iterrows() 
                          if row['sentiment'] == 'Negative')
    
    neutral_weighted = sum(1 + row['like_count'] 
                         for _, row in results_df.iterrows() 
                         if row['sentiment'] == 'Neutral')
    
    total_weighted = positive_weighted + negative_weighted + neutral_weighted
    
    # Calculate percentages
    positive_pct = (positive_weighted / total_weighted) * 100 if total_weighted > 0 else 0
    negative_pct = (negative_weighted / total_weighted) * 100 if total_weighted > 0 else 0
    neutral_pct = (neutral_weighted / total_weighted) * 100 if total_weighted > 0 else 0
    
    return positive_pct, negative_pct, neutral_pct, total_weighted

def calculate_unweighted_percentages(results_df):
    """Calculate standard unweighted sentiment percentages"""
    if results_df.empty:
        return 0, 0, 0
    
    total = len(results_df)
    positive = len(results_df[results_df['sentiment'] == 'Positive'])
    negative = len(results_df[results_df['sentiment'] == 'Negative'])
    neutral = len(results_df[results_df['sentiment'] == 'Neutral'])
    
    positive_pct = (positive / total) * 100
    negative_pct = (negative / total) * 100
    neutral_pct = (neutral / total) * 100
    
    return positive_pct, negative_pct, neutral_pct

def main():
    # Load models
    model, tfidf_vectorizer, label_encoder, tokenizer, model_info, max_length = load_models()

    if model is None:
        st.error("Could not load models. Please run the training notebook first.")
        return

    # Header
    st.markdown("""
    <div class="header-container">
       <h1 class="header-title" style="margin: 0 !important; padding: 0 !important; color: #000000;text-align: center;font-size: 1.8rem;font-weight: 700;">YouTube Sentiment Analyzer</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content container
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Input section
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        video_url = st.text_input("YouTube Video URL", 
                                placeholder="https://www.youtube.com/watch?v=...",
                                key="youtube_url")
        analyze_btn = st.button("Analyze Video", key="analyze_video", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content columns
    left_col, right_col = st.columns([1, 2])

    # LEFT COLUMN - Video Info & Sentiment Summary
    with left_col:
        # Video Information Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Video Information</div>', unsafe_allow_html=True)
        
        if st.session_state.video_info:
            video_info = st.session_state.video_info
            st.markdown(f"""
            <div class="video-info">
                <div class="video-thumbnail">
                    <img src="{video_info['thumbnail_url']}" alt="Video thumbnail">
                </div>
                <div class="video-details">
                    <div class="video-title">{video_info['title']}</div>
                    <div class="video-meta">{video_info['channel_title']}</div>
                    <div class="video-meta">{video_info['published_at']}</div>
                    <div class="video-stats">
                        <span>👀 {video_info['view_count']}</span>
                        <span>👍 {video_info['like_count']}</span>
                        <span>💬 {video_info['comment_count']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                Enter a YouTube URL to see video details
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Sentiment Analysis Results
        if st.session_state.analysis_results is not None:
            results_df = st.session_state.analysis_results

            total_comments = len(results_df)
            

            
            # Calculate both weighted and unweighted percentages
            pos_w, neg_w, neu_w, total_weighted = calculate_weighted_percentages(results_df)
            pos_uw, neg_uw, neu_uw = calculate_unweighted_percentages(results_df)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Sentiment Analysis (Weighted by Likes)</div>', unsafe_allow_html=True)
            
            # Display weighted percentages
            st.markdown(f"""
            <div class="sentiment-grid">
                <div class="sentiment-item positive">
                    <div class="sentiment-value">{pos_w:.1f}%</div>
                    <div class="sentiment-label">Positive</div>
                </div>
                <div class="sentiment-item neutral">
                    <div class="sentiment-value">{neu_w:.1f}%</div>
                    <div class="sentiment-label">Neutral</div>
                </div>
                <div class="sentiment-item negative">
                    <div class="sentiment-value">{neg_w:.1f}%</div>
                    <div class="sentiment-label">Negative</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show comparison with unweighted percentages
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 0.8rem; background: #f0f4f8; border-radius: 6px;">
                <div style="font-size: 0.85rem; color: #4a5568; text-align: center;">
                    <b>Standard Percentages:</b> 
                    Positive: {pos_uw:.1f}% | Neutral: {neu_uw:.1f}% | Negative: {neg_uw:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
           #  Show comment count info
            st.markdown(f"""
            <div style="margin-bottom: 1rem; padding: 0.5rem; background: #f0f4f8; border-radius: 6px;">
                <div style="font-size: 0.85rem; color: #4a5568; text-align: center;">
                    Analyzed {total_comments} comments
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if 'gating_stats' in st.session_state:
                gs = st.session_state.gating_stats
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 0.8rem; background: #f0f4f8; border-radius: 6px;">
                    <div style="font-size: 0.85rem; color: #4a5568; text-align: center;">
                        <b>Model Collaboration:</b> 
                        SVM corrected BiLSTM in {gs['svm_overrides']} cases ({gs['override_pct']:.1f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT COLUMN - Comments
    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Top Comments</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_results is not None:
            results_df = st.session_state.analysis_results
            
            # Three columns for different sentiments
            comment_col1, comment_col2, comment_col3 = st.columns(3)
            sentiment_types = ['Positive', 'Negative', 'Neutral']
            columns = [comment_col1, comment_col2, comment_col3]
            
            for i, sentiment_type in enumerate(sentiment_types):
                with columns[i]:
                    sentiment_df = results_df[results_df['sentiment'] == sentiment_type]
                    
                    st.markdown(f"""
                    <div class="comment-section-header {sentiment_type.lower()}">
                        {sentiment_type} ({len(sentiment_df)})
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show top 5 comments for each sentiment
                    for _, row in sentiment_df.head(5).iterrows():
                        sentiment_class = row['sentiment'].lower()
                        comment_text = row['comment'][:120] + ('...' if len(row['comment']) > 120 else '')
                        author_name = row['author'][:20] + ('...' if len(row['author']) > 20 else '')
                        
                        st.markdown(f"""
                        <div class="comment-item {sentiment_class}">
                            <div class="comment-text">"{comment_text}"</div>
                            <div class="comment-meta">
                                <span>👤 {author_name}</span>
                                <span>👍 {row['like_count']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    if len(sentiment_df) == 0:
                        st.markdown(f'<div class="empty-state">No {sentiment_type.lower()} comments</div>', 
                                   unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                Analyze a YouTube video to see comments
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-content

    # Analysis Logic
    if analyze_btn and video_url:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL format")
            return

        with st.spinner("Analyzing video... (this may take a while for videos with many comments)"):
            # Get video info
            video_info = get_video_info(video_id)
            if not video_info:
                st.error("Could not fetch video information")
                return
            
            st.session_state.video_info = video_info

            # Get comments
            comments_data = fetch_youtube_comments(video_id)
            if not comments_data:
                st.error("Could not fetch comments")
                return
            
            st.info(f"Fetched {len(comments_data)} comments for analysis")
            st.session_state.comments_data = comments_data

            # Analyze sentiment
            comments = [item['comment'] for item in comments_data]
            sentiments, confidence_scores = predict_sentiment(
                comments, model, tfidf_vectorizer, label_encoder,
                tokenizer, model_info, max_length
            )

            # Store results
            results_df = pd.DataFrame({
                'comment': comments,
                'sentiment': sentiments,
                'confidence': confidence_scores if confidence_scores is not None else [0.0] * len(comments),
                'author': [item['author'] for item in comments_data],
                'like_count': [item['like_count'] for item in comments_data],
                'published_at': [item['published_at'] for item in comments_data]
            })
            
            st.session_state.analysis_results = results_df
            st.rerun()

if __name__ == "__main__":
    main()




# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import re
# import pickle
# import joblib
# import requests
# import json
# from datetime import datetime
# import time
# import os
# import warnings
# warnings.filterwarnings('ignore')

# # Deep learning imports
# try:
#     import tensorflow as tf
#     from tensorflow.keras.models import load_model
#     from tensorflow.keras.preprocessing.sequence import pad_sequences
#     DEEP_LEARNING_AVAILABLE = True
# except ImportError:
#     DEEP_LEARNING_AVAILABLE = False
#     st.warning("TensorFlow not available. Deep learning models will not work.")

# # YouTube Data API configuration
# YOUTUBE_API_KEY = 'AIzaSyA-oQaAVmJBL43ar6rLxkYgOdyOFcBHEy0'
# YOUTUBE_API_BASE_URL = 'https://www.googleapis.com/youtube/v3'

# # Set page config at the very top
# st.set_page_config(
#     page_title="YouTube Sentiment Analyzer",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # Modern, professional CSS styling (adjusted for compactness)
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
# html, body, #root, .stApp {
#         font-family: 'Inter', sans-serif;
#         background-color: #f8fafc;
#         margin: 0 !important;
#         padding: 0 !important;
#         height: 100%;
#     }
    
#     /* HEADER - ABSOLUTE TOP */
#     .header-container {
#         position: relative;
#         padding: 0.8rem 1rem;
#         background: white;
#         border-bottom: 1px solid #e2e8f0;
#         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
#         margin-top: -1.5rem !important; /* Force pull up */
#         top: -8px !important; /* Additional pull up */
#     }
    
#     .header-title {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #1a202c;
#         margin: 0 !important;
#         padding: 0 !important;
#         line-height: 1;
#     }
    
#     /* REMOVE ALL STREAMLIT PADDING */
#     .stApp > div {
#         padding: 0 !important;
#     }
    
#     .main-content {
#         padding-top: 0 !important;
#         margin-top: 0 !important;
#     }
    
#     /* FORCE HEADER TO TOP */
#     .stApp > header {
#         display: none !important;
#     }

#     /* Input fields */
#     .stTextInput > div > div > input,
#     .stTextArea > div > div > textarea, 
#     .stSelectbox > div > div > div {
#         border-radius: 6px !important;
#         border: 1px solid #e2e8f0 !important;
#         padding: 0.5rem 0.6rem !important; /* Reduced padding */
#         font-size: 0.8rem !important; /* Smaller font */
#         height: 40px !important; /* Reduced height */
#         min-width: 100%;
#     }
    
#     /* Textarea specific */
#     .stTextArea > div > div > textarea {
#         line-height: 1.5 !important;
#         resize: none !important;
#     }
    
#     /* Input labels */
#     .stTextInput > label,
#     .stTextArea > label,
#     .stSelectbox > label {
#         font-size: 0.8rem !important; /* Smaller font */
#         font-weight: 500 !important;
#         color: #2d3748 !important;
#         margin-bottom: 0.25rem !important; /* Reduced margin */
#     }
    
#     /* Buttons */
#     .stButton > button {
#         background-color: #1a73e8 !important;
#         color: white !important;
#         border: none !important;
#         border-radius: 6px !important;
#         padding: 0.5rem 1rem !important; /* Reduced padding */
#         font-weight: 500 !important;
#         font-size: 0.85rem !important; /* Slightly smaller font */
#         height: 40px !important; /* Reduced height */
#         min-width: 120px !important; /* Slightly narrower button */
#         transition: all 0.2s !important;
#     }
    
#     .stButton > button:hover {
#         background-color: #1765cc !important;
#         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
#         transform: translateY(-1px) !important;
#     }
    
#     /* Cards */
#     .card {
#         background: white;
#         border-radius: 8px;
#         padding: 1rem; /* Reduced padding */
#         margin-bottom: 0.8rem; /* Reduced margin */
#         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
#         border: 1px solid #e2e8f0;
#         height: 100%;
#     }
    
#     .card-header {
#         font-size: 1rem; /* Smaller font */
#         font-weight: 600;
#         color: #2d3748;
#         margin-bottom: 0.75rem; /* Reduced margin */
#         padding-bottom: 0.5rem; /* Reduced padding */
#         border-bottom: 1px solid #e2e8f0;
#     }
    
#     /* Video info */
#     .video-info {
#         display: flex;
#         gap: 0.8rem; /* Reduced gap */
#         margin-bottom: 0.8rem; /* Reduced margin */
#     }
    
#     .video-thumbnail {
#         flex-shrink: 0;
#         width: 140px; /* Slightly smaller thumbnail */
#         height: 78px; /* Adjusted height for aspect ratio */
#         border-radius: 6px;
#         overflow: hidden;
#     }
    
#     .video-thumbnail img {
#         width: 100%;
#         height: 100%;
#         object-fit: cover;
#     }
    
#     .video-details {
#         flex: 1;
#     }
    
#     .video-title {
#         font-size: 0.95rem; /* Slightly smaller font */
#         font-weight: 600;
#         color: #1a202c;
#         margin-bottom: 0.4rem; /* Reduced margin */
#         line-height: 1.3;
#     }
    
#     .video-meta {
#         font-size: 0.8rem; /* Smaller font */
#         color: #4a5568;
#         margin-bottom: 0.2rem; /* Reduced margin */
#     }
    
#     .video-stats {
#         display: flex;
#         gap: 0.8rem; /* Reduced gap */
#         font-size: 0.8rem; /* Smaller font */
#         color: #4a5568;
#     }
    
#     /* Sentiment grid */
#     .sentiment-grid {
#         display: grid;
#         grid-template-columns: repeat(3, 1fr);
#         gap: 0.8rem; /* Reduced gap */
#         margin-bottom: 0.8rem; /* Reduced margin */
#     }
    
#     .sentiment-item {
#         background: #f8fafc;
#         border-radius: 6px;
#         padding: 0.8rem; /* Reduced padding */
#         text-align: center;
#         border-left: 3px solid #e2e8f0;
#     }
    
#     .sentiment-item.positive { border-left-color: #38a169; }
#     .sentiment-item.negative { border-left-color: #e53e3e; }
#     .sentiment-item.neutral { border-left-color: #d69e2e; }
    
#     .sentiment-value {
#         font-size: 1.2rem; /* Smaller font */
#         font-weight: 700;
#         margin: 0.4rem 0; /* Reduced margin */
#         color: #4a5568;

#     }
    
#     .sentiment-label {
#         font-size: 0.8rem; /* Smaller font */
#         color: #4a5568;
#     }
    
#     /* Comments */
#     .comments-container {
#         padding-right: 0.5rem; 
#     }
    
#     .comment-section-header {
#         font-size: 0.85rem; /* Smaller font */
#         font-weight: 600;
#         color: white;
#         padding: 0.4rem; /* Reduced padding */
#         border-radius: 4px;
#         margin-bottom: 0.6rem; /* Reduced margin */
#         text-align: center;
#     }
    
#     .comment-section-header.positive {
#         background: #38a169;
#     }
    
#     .comment-section-header.negative {
#         background: #e53e3e;
#     }
#     .comment-section-header.neutral {
#         background: #d69e2e;
#     }
    
#     .comment-item {
#         background: #f8fafc;
#         border-radius: 6px;
#         padding: 0.8rem; /* Reduced padding */
#         margin-bottom: 0.5rem; /* Reduced margin */
#         border-left: 3px solid #e2e8f0;
#     }
    
#     .comment-text {
#         font-size: 0.85rem; /* Smaller font */
#         line-height: 1.4; /* Slightly reduced line height */
#         color: #2d3748;
#         margin-bottom: 0.4rem; /* Reduced margin */
#     }
    
#     .comment-meta {
#         font-size: 0.75rem; /* Smaller font */
#         color: #718096;
#         display: flex;
#         justify-content: space-between;
#     }
    
#     /* Empty state */
#     .empty-state {
#         text-align: center;
#         color: #718096;
#         font-size: 0.85rem; /* Smaller font */
#         padding: 1.5rem 0.8rem; /* Reduced padding */
#         background: #f8fafc;
#         border-radius: 6px;
#         border: 1px dashed #cbd5e0;
#     }
    
#     /* Hide Streamlit elements */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     .stDeployButton {visibility: hidden;}
    
#     /* Responsive adjustments */
#     @media (max-width: 768px) {
#         .header-title {
#             font-size: 1.8rem;
#         }
        
#         .video-info {
#             flex-direction: column;
#         }
        
#         .video-thumbnail {
#             width: 100%;
#             height: 160px;
#         }
        
#         .sentiment-grid {
#             grid-template-columns: 1fr;
#         }
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'comments_data' not in st.session_state:
#     st.session_state.comments_data = None
# if 'analysis_results' not in st.session_state:
#     st.session_state.analysis_results = None
# if 'video_info' not in st.session_state:
#     st.session_state.video_info = None
# if 'custom_text_result' not in st.session_state:
#     st.session_state.custom_text_result = None

# @st.cache_resource
# def load_models():
#     """Load trained models and preprocessing components for Smart Gating Ensemble"""
#     try:
#         # Load base components
#         tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
#         label_encoder = joblib.load('models/label_encoder.pkl')
        
#         # Load the complete ensemble package
#         ensemble_package = joblib.load('models/smart_gating_ensemble.pkl')
        
#         # Verify all required components exist
#         required_components = [
#             'svm_model', 'bilstm_model', 'gating_classifier',
#             'tokenizer', 'max_sequence_length'
#         ]
        
#         if not all(key in ensemble_package for key in required_components):
#             st.error("Missing components in ensemble package")
#             return None, None, None, None, None, None
            
#         return (
#             ensemble_package,  # model
#             tfidf_vectorizer,
#             label_encoder,
#             ensemble_package['tokenizer'],
#             {'best_model_type': 'smart_gating'},
#             ensemble_package['max_sequence_length']
#         )

#     except Exception as e:
#         st.error(f"Error loading models: {str(e)}")
#         st.stop()
#         return None, None, None, None, None, None
# # def load_models():
# #     """Load trained models and preprocessing components"""
# #     try:
# #         tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
# #         label_encoder = joblib.load('models/label_encoder.pkl')

# #         with open('models/model_info.txt', 'r') as f:
# #             model_info = {}
# #             for line in f:
# #                 key, value = line.strip().split(': ')
# #                 model_info[key] = value

# #         if model_info['best_model_type'] == 'deep_learning' and DEEP_LEARNING_AVAILABLE:
# #             model = load_model('models/best_model.h5')
# #             with open('models/tokenizer.pkl', 'rb') as f:
# #                 tokenizer = pickle.load(f)
# #             max_length = int(model_info['max_length'])
# #             return model, tfidf_vectorizer, label_encoder, tokenizer, model_info, max_length
# #         else:
# #             model = joblib.load('models/best_model.pkl')
# #             return model, tfidf_vectorizer, label_encoder, None, model_info, None

# #     except Exception as e:
# #         st.error(f"Error loading models: {str(e)}")
# #         st.info("Please make sure you have run the model training notebook first.")
# #         return None, None, None, None, None, None

# def preprocess_text(text):
#     """Preprocess text for sentiment analysis"""
#     if pd.isna(text):
#         return ""
    
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     text = re.sub(r'@\w+|#\w+', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
#     text = re.sub(r'\d+', '', text)
#     return text.strip()

# def extract_video_id(url):
#     """Extract video ID from YouTube URL"""
#     patterns = [
#         r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
#         r'(?:embed\/)([0-9A-Za-z_-]{11})',
#         r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
#         r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
#     ]
    
#     for pattern in patterns:
#         match = re.search(pattern, url)
#         if match:
#             return match.group(1)
#     return None

# def get_video_info(video_id):
#     """Fetch video information from YouTube Data API"""
#     try:
#         url = f"{YOUTUBE_API_BASE_URL}/videos"
#         params = {
#             'part': 'snippet,statistics',
#             'id': video_id,
#             'key': YOUTUBE_API_KEY
#         }

#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         data = response.json()

#         if not data['items']:
#             return None

#         video = data['items'][0]
#         snippet = video['snippet']
#         statistics = video['statistics']

#         def format_number(num_str):
#             try:
#                 num = int(num_str)
#                 if num >= 1000000:
#                     return f"{num/1000000:.1f}M"
#                 elif num >= 1000:
#                     return f"{num/1000:.1f}K"
#                 else:
#                     return str(num)
#             except:
#                 return num_str

#         return {
#             'title': snippet['title'],
#             'description': snippet['description'][:150] + "..." if len(snippet['description']) > 150 else snippet['description'],
#             'channel_title': snippet['channelTitle'],
#             'published_at': snippet['publishedAt'][:10],
#             'thumbnail_url': snippet['thumbnails']['high']['url'],
#             'view_count': format_number(statistics.get('viewCount', '0')),
#             'like_count': format_number(statistics.get('likeCount', '0')),
#             'comment_count': format_number(statistics.get('commentCount', '0')),
#         }

#     except Exception as e:
#         st.error(f"Error fetching video info: {str(e)}")
#         return None

# def fetch_youtube_comments(video_id, max_comments=200):
#     """Fetch YouTube comments using YouTube Data API"""
#     try:
#         comments_data = []
#         next_page_token = None

#         while len(comments_data) < max_comments:
#             url = f"{YOUTUBE_API_BASE_URL}/commentThreads"
#             params = {
#                 'part': 'snippet',
#                 'videoId': video_id,
#                 'key': YOUTUBE_API_KEY,
#                 'maxResults': min(100, max_comments - len(comments_data)),
#                 'order': 'relevance'
#             }

#             if next_page_token:
#                 params['pageToken'] = next_page_token

#             response = requests.get(url, params=params)
#             response.raise_for_status()
#             data = response.json()

#             if not data.get('items'):
#                 break

#             for item in data['items']:
#                 comment = item['snippet']['topLevelComment']['snippet']
#                 comments_data.append({
#                     'comment': comment['textDisplay'],
#                     'author': comment['authorDisplayName'],
#                     'published_at': comment['publishedAt'][:10],
#                     'like_count': comment['likeCount']
#                 })

#             next_page_token = data.get('nextPageToken')
#             if not next_page_token:
#                 break

#             time.sleep(0.1)

#         return comments_data[:max_comments]

#     except Exception as e:
#         st.error(f"Error fetching comments: {str(e)}")
#         return []

# # def predict_sentiment(comments, model, tfidf_vectorizer, label_encoder, tokenizer, model_info, max_length):
# #     processed_comments = [preprocess_text(comment) for comment in comments]
# #     sentiments = []
# #     confidence_scores = []

# #     if model_info.get('best_model_type') == 'deep_learning':
# #         if not DEEP_LEARNING_AVAILABLE:
# #             st.error("Deep learning model selected but TensorFlow not available. Cannot predict sentiment.")
# #             return ["Error"] * len(comments), [0.0] * len(comments)

# #         if tokenizer is None:
# #             st.error("Tokenizer not loaded for deep learning model. Cannot predict sentiment.")
# #             return ["Error"] * len(comments), [0.0] * len(comments)

# #         sequences = tokenizer.texts_to_sequences(processed_comments)
# #         padded_sequences = pad_sequences(sequences, maxlen=max_length)
# #         predictions_proba = model.predict(padded_sequences)
# #         predicted_classes = np.argmax(predictions_proba, axis=1)
# #         confidence_scores = np.max(predictions_proba, axis=1)
# #         sentiments = label_encoder.inverse_transform(predicted_classes)
# #         return sentiments, confidence_scores

# #     elif model_info.get('best_model_type') == 'ensemble':
# #         # --- NEW LOGIC FOR ENSEMBLE MODEL ---
# #         if not isinstance(model, dict):
# #             st.error("Ensemble model not loaded correctly. Expected a dictionary.")
# #             return ["Error"] * len(comments), [0.0] * len(comments)

# #         svm_model = model.get('svm_model')
# #         bilstm_model = model.get('bilstm_model')
# #         svm_weight = model.get('svm_weight')
# #         bilstm_weight = model.get('bilstm_weight')
# #         ensemble_tokenizer = model.get('tokenizer') # Use the tokenizer from the ensemble package
# #         ensemble_max_length = model.get('max_sequence_length') # Use max_length from ensemble package

# #         if svm_model is None or bilstm_model is None or ensemble_tokenizer is None:
# #             st.error("Missing components in ensemble model. Cannot predict sentiment.")
# #             return ["Error"] * len(comments), [0.0] * len(comments)

# #         # SVM Prediction
# #         svm_features = tfidf_vectorizer.transform(processed_comments)
# #         svm_predictions_proba = svm_model.predict_proba(svm_features)

# #         # BiLSTM Prediction
# #         if not DEEP_LEARNING_AVAILABLE:
# #             st.warning("TensorFlow not available for BiLSTM part of ensemble. Only SVM predictions will be used.")
# #             # Fallback to only SVM if DL is not available for BiLSTM part
# #             combined_predictions_proba = svm_predictions_proba
# #         else:
# #             dl_sequences = ensemble_tokenizer.texts_to_sequences(processed_comments)
# #             padded_sequences_dl = pad_sequences(dl_sequences, maxlen=ensemble_max_length)
# #             bilstm_predictions_proba = bilstm_model.predict(padded_sequences_dl)

# #             # Combine predictions
# #             combined_predictions_proba = (svm_predictions_proba * svm_weight) + \
# #                                          (bilstm_predictions_proba * bilstm_weight)

# #         predicted_classes = np.argmax(combined_predictions_proba, axis=1)
# #         confidence_scores = np.max(combined_predictions_proba, axis=1)
# #         sentiments = label_encoder.inverse_transform(predicted_classes)
# #         return sentiments, confidence_scores
# #         # --- END NEW LOGIC ---

# #     else: # This block will now primarily handle the 'svm' best_model_type
# #         features = tfidf_vectorizer.transform(processed_comments)
# #         predictions = model.predict(features)
# #         if hasattr(model, 'predict_proba'):
# #             predictions_proba = model.predict_proba(features)
# #             confidence_scores = np.max(predictions_proba, axis=1)
# #         else:
# #             confidence_scores = None
# #         predicted_classes = predictions
# #         sentiments = label_encoder.inverse_transform(predicted_classes)
# #         return sentiments, confidence_scores
# def predict_sentiment(comments, model, tfidf_vectorizer, label_encoder, tokenizer, model_info, max_length):
#     """Predict sentiment using Smart Gating Ensemble"""
#     processed_comments = [preprocess_text(comment) for comment in comments]
#     sentiments = []
#     confidence_scores = []
#     gating_decisions = []  # Track when SVM overrides BiLSTM

#     if model_info.get('best_model_type') != 'smart_gating':
#         st.error("Incorrect model type loaded")
#         return ["Error"] * len(comments), [0.0] * len(comments)

#     # Unpack ensemble components
#     svm_model = model['svm_model']
#     bilstm_model = model['bilstm_model']
#     gate_clf = model['gating_classifier']
#     ensemble_tokenizer = model['tokenizer']
#     ensemble_max_length = model['max_sequence_length']

#     # 1. Get SVM predictions
#     svm_features = tfidf_vectorizer.transform(processed_comments)
#     svm_probs = svm_model.predict_proba(svm_features)
#     svm_preds = np.argmax(svm_probs, axis=1)

#     # 2. Get BiLSTM predictions
#     if not DEEP_LEARNING_AVAILABLE:
#         st.error("TensorFlow required for BiLSTM predictions")
#         return ["Error"] * len(comments), [0.0] * len(comments)
        
#     sequences = ensemble_tokenizer.texts_to_sequences(processed_comments)
#     padded_sequences = pad_sequences(sequences, maxlen=ensemble_max_length)
#     bilstm_probs = bilstm_model.predict(padded_sequences)
#     bilstm_preds = np.argmax(bilstm_probs, axis=1)

#     # 3. Create meta-features for gating
#     meta_features = np.hstack([svm_probs, bilstm_probs])

#     # 4. Make final predictions
#     for i in range(len(processed_comments)):
#         default_pred = bilstm_preds[i]  # Start with BiLSTM
        
#         # Check if models disagree
#         if svm_preds[i] != bilstm_preds[i]:
#             # Ask gating classifier which model to trust
#             gate_pred = gate_clf.predict(meta_features[i:i+1])[0]
            
#             if gate_pred == svm_preds[i]:  # Gate agrees with SVM
#                 final_pred = svm_preds[i]
#                 gating_decisions.append(True)  # SVM was used
#             else:
#                 final_pred = default_pred
#                 gating_decisions.append(False)
#         else:
#             final_pred = default_pred
#             gating_decisions.append(False)
        
#         # Get confidence from the chosen model
#         if final_pred == svm_preds[i]:
#             confidence = svm_probs[i][final_pred]
#         else:
#             confidence = bilstm_probs[i][final_pred]
            
#         sentiments.append(label_encoder.inverse_transform([final_pred])[0])
#         confidence_scores.append(confidence)

#     # Store gating stats in session state for visualization
#     st.session_state.gating_stats = {
#         'total': len(gating_decisions),
#         'svm_overrides': sum(gating_decisions),
#         'override_pct': (sum(gating_decisions) / len(gating_decisions)) * 100
#     }
    
#     return sentiments, confidence_scores

#     # Fallback in case none of the conditions are met (shouldn't happen with the above structure)
#     return ["Error"] * len(comments), [0.0] * len(comments)
# def main():
#     # Load models
#     model, tfidf_vectorizer, label_encoder, tokenizer, model_info, max_length = load_models()

#     if model is None:
#         st.error("Could not load models. Please run the training notebook first.")
#         return

#     # Header at the very top

#     st.markdown("""
#     <div class="header-container">
#         <h1 class="header-title" style="margin: 0 !important; padding: 0 !important; color: #000000;text-align: center;font-size: 1.8rem;font-weight: 700;">YouTube Sentiment Analyzer</h1>
#     </div>
#     """, unsafe_allow_html=True)
#     # Main content container
#     st.markdown('<div class="main-content">', unsafe_allow_html=True)

#     # Input section
#     with st.container():
#         st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
#         # Single column layout for inputs, decreased width of col1
#         col1, col2 = st.columns([1.5, 1]) 
        
#         with col1:
#             video_url = st.text_input("YouTube Video URL", 
#                                     placeholder="https://www.youtube.com/watch?v=...",
#                                     key="youtube_url")
        
#         with col2:
#             comment_count = st.selectbox("Number of Comments", 
#                                        [50, 100, 200, 300],
#                                        index=1)
        
#         # Single button row
#         analyze_btn = st.button("Analyze Video", key="analyze_video")
        
#         st.markdown('</div>', unsafe_allow_html=True)

#     # Main content columns
#     left_col, right_col = st.columns([1, 2])

#     # LEFT COLUMN - Video Info & Sentiment Summary
#     with left_col:
#         # Video Information Card
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown('<div class="card-header">Video Information</div>', unsafe_allow_html=True)
        
#         if st.session_state.video_info:
#             video_info = st.session_state.video_info
#             st.markdown(f"""
#             <div class="video-info">
#                 <div class="video-thumbnail">
#                     <img src="{video_info['thumbnail_url']}" alt="Video thumbnail">
#                 </div>
#                 <div class="video-details">
#                     <div class="video-title">{video_info['title']}</div>
#                     <div class="video-meta">{video_info['channel_title']}</div>
#                     <div class="video-meta">{video_info['published_at']}</div>
#                     <div class="video-stats">
#                         <span>👀 {video_info['view_count']}</span>
#                         <span>👍 {video_info['like_count']}</span>
#                         <span>💬 {video_info['comment_count']}</span>
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown("""
#             <div class="empty-state">
#                 Enter a YouTube URL to see video details
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)

#         # Sentiment Analysis Results
#         if st.session_state.analysis_results is not None:
#             st.markdown('<div class="card">', unsafe_allow_html=True)
#             st.markdown('<div class="card-header">Sentiment Analysis</div>', unsafe_allow_html=True)
            
#             results_df = st.session_state.analysis_results
#             total_comments = len(results_df)
#             if 'gating_stats' in st.session_state:
#                 gs = st.session_state.gating_stats
#                 st.markdown(f"""
#                 <div style="margin-top: 1rem; padding: 0.8rem; background: #f0f4f8; border-radius: 6px;">
#                     <div style="font-size: 0.85rem; color: #4a5568;">
#                         <b>Model Collaboration:</b> 
#                         SVM corrected BiLSTM in {gs['svm_overrides']}/{gs['total']} 
#                         cases ({gs['override_pct']:.1f}%)
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             st.markdown('</div>', unsafe_allow_html=True)
#             if total_comments > 0:
#                 positive_count = len(results_df[results_df['sentiment'] == 'Positive'])
#                 negative_count = len(results_df[results_df['sentiment'] == 'Negative'])
#                 neutral_count = len(results_df[results_df['sentiment'] == 'Neutral'])

#                 positive_pct = (positive_count / total_comments) * 100
#                 negative_pct = (negative_count / total_comments) * 100
#                 neutral_pct = (neutral_count / total_comments) * 100

#                 st.markdown(f"""
#                 <div class="sentiment-grid">
#                     <div class="sentiment-item positive">
#                         <div class="sentiment-value">{positive_pct:.1f}%</div>
#                         <div class="sentiment-label">Positive</div>
#                     </div>
#                     <div class="sentiment-item neutral">
#                         <div class="sentiment-value">{neutral_pct:.1f}%</div>
#                         <div class="sentiment-label">Neutral</div>
#                     </div>
#                     <div class="sentiment-item negative">
#                         <div class="sentiment-value">{negative_pct:.1f}%</div>
#                         <div class="sentiment-label">Negative</div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             st.markdown('</div>', unsafe_allow_html=True)

#     # RIGHT COLUMN - Comments
#     with right_col:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown('<div class="card-header">Top Comments</div>', unsafe_allow_html=True)
        
#         if st.session_state.analysis_results is not None:
#             results_df = st.session_state.analysis_results
            
#             # Three columns for different sentiments
#             comment_col1, comment_col2, comment_col3 = st.columns(3)
#             sentiment_types = ['Positive', 'Negative', 'Neutral']
#             columns = [comment_col1, comment_col2, comment_col3]
            
#             for i, sentiment_type in enumerate(sentiment_types):
#                 with columns[i]:
#                     sentiment_df = results_df[results_df['sentiment'] == sentiment_type]
                    
#                     st.markdown(f"""
#                     <div class="comment-section-header {sentiment_type.lower()}">
#                         {sentiment_type} ({len(sentiment_df)})
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     for _, row in sentiment_df.head(3).iterrows():  # Show max 3 comments
#                         sentiment_class = row['sentiment'].lower()
#                         comment_text = row['comment'][:120] + ('...' if len(row['comment']) > 120 else '')
#                         author_name = row['author'][:20] + ('...' if len(row['author']) > 20 else '')
                        
#                         st.markdown(f"""
#                         <div class="comment-item {sentiment_class}">
#                             <div class="comment-text">"{comment_text}"</div>
#                             <div class="comment-meta">
#                                 <span>👤 {author_name}</span>
#                                 <span>👍 {row['like_count']}</span>
#                             </div>
#                         </div>
#                         """, unsafe_allow_html=True)
#                     if len(sentiment_df) == 0:
#                         st.markdown(f'<div class="empty-state">No {sentiment_type.lower()} comments</div>', 
#                                    unsafe_allow_html=True)
#         else:
#             st.markdown("""
#             <div class="empty-state">
#                 Analyze a YouTube video to see comments
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)  # Close main-content

#     # Analysis Logic
#     if analyze_btn and video_url:
#         video_id = extract_video_id(video_url)
#         if not video_id:
#             st.error("Invalid YouTube URL format")
#             return

#         with st.spinner("Analyzing video..."):
#             # Get video info
#             video_info = get_video_info(video_id)
#             if not video_info:
#                 st.error("Could not fetch video information")
#                 return
            
#             st.session_state.video_info = video_info

#             # Get comments
#             comments_data = fetch_youtube_comments(video_id, comment_count)
#             if not comments_data:
#                 st.error("Could not fetch comments")
#                 return
            
#             st.session_state.comments_data = comments_data

#             # Analyze sentiment
#             comments = [item['comment'] for item in comments_data]
#             sentiments, confidence_scores = predict_sentiment(
#                 comments, model, tfidf_vectorizer, label_encoder,
#                 tokenizer, model_info, max_length
#             )

#             # Store results
#             results_df = pd.DataFrame({
#                 'comment': comments,
#                 'sentiment': sentiments,
#                 'confidence': confidence_scores if confidence_scores is not None else [0.0] * len(comments),
#                 'author': [item['author'] for item in comments_data],
#                 'like_count': [item['like_count'] for item in comments_data],
#                 'published_at': [item['published_at'] for item in comments_data]
#             })
            
#             st.session_state.analysis_results = results_df
#             st.session_state.custom_text_result = None
#             st.rerun()

# if __name__ == "__main__":
#     main()







# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import re
# import pickle
# import joblib
# import requests
# import json
# from datetime import datetime
# import time
# import os
# import warnings
# warnings.filterwarnings('ignore')

# # Deep learning imports
# try:
#     import tensorflow as tf
#     from tensorflow.keras.models import load_model
#     from tensorflow.keras.preprocessing.sequence import pad_sequences
#     DEEP_LEARNING_AVAILABLE = True
# except ImportError:
#     DEEP_LEARNING_AVAILABLE = False
#     st.warning("TensorFlow not available. Deep learning models will not work.")

# # YouTube Data API configuration
# YOUTUBE_API_KEY = 'AIzaSyA-oQaAVmJBL43ar6rLxkYgOdyOFcBHEy0'
# YOUTUBE_API_BASE_URL = 'https://www.googleapis.com/youtube/v3'

# # Set page config at the very top
# st.set_page_config(
#     page_title="YouTube Sentiment Analyzer",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # Modern, professional CSS styling (adjusted for compactness)
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
# html, body, #root, .stApp {
#         font-family: 'Inter', sans-serif;
#         background-color: #f8fafc;
#         margin: 0 !important;
#         padding: 0 !important;
#         height: 100%;
#     }
    
#     /* HEADER - ABSOLUTE TOP */
#     .header-container {
#         position: relative;
#         padding: 0.8rem 1rem;
#         background: white;
#         border-bottom: 1px solid #e2e8f0;
#         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
#         margin-top: -1.5rem !important; /* Force pull up */
#         top: -8px !important; /* Additional pull up */
#     }
    
#     .header-title {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #1a202c;
#         margin: 0 !important;
#         padding: 0 !important;
#         line-height: 1;
#     }
    
#     /* REMOVE ALL STREAMLIT PADDING */
#     .stApp > div {
#         padding: 0 !important;
#     }
    
#     .main-content {
#         padding-top: 0 !important;
#         margin-top: 0 !important;
#     }
    
#     /* FORCE HEADER TO TOP */
#     .stApp > header {
#         display: none !important;
#     }

#     /* Input fields */
#     .stTextInput > div > div > input,
#     .stTextArea > div > div > textarea, 
#     .stSelectbox > div > div > div {
#         border-radius: 6px !important;
#         border: 1px solid #e2e8f0 !important;
#         padding: 0.5rem 0.6rem !important; /* Reduced padding */
#         font-size: 0.8rem !important; /* Smaller font */
#         height: 40px !important; /* Reduced height */
#         min-width: 100%;
#     }
    
#     /* Textarea specific */
#     .stTextArea > div > div > textarea {
#         line-height: 1.5 !important;
#         resize: none !important;
#     }
    
#     /* Input labels */
#     .stTextInput > label,
#     .stTextArea > label,
#     .stSelectbox > label {
#         font-size: 0.8rem !important; /* Smaller font */
#         font-weight: 500 !important;
#         color: #2d3748 !important;
#         margin-bottom: 0.25rem !important; /* Reduced margin */
#     }
    
#     /* Buttons */
#     .stButton > button {
#         background-color: #1a73e8 !important;
#         color: white !important;
#         border: none !important;
#         border-radius: 6px !important;
#         padding: 0.5rem 1rem !important; /* Reduced padding */
#         font-weight: 500 !important;
#         font-size: 0.85rem !important; /* Slightly smaller font */
#         height: 40px !important; /* Reduced height */
#         min-width: 120px !important; /* Slightly narrower button */
#         transition: all 0.2s !important;
#     }
    
#     .stButton > button:hover {
#         background-color: #1765cc !important;
#         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
#         transform: translateY(-1px) !important;
#     }
    
#     /* Cards */
#     .card {
#         background: white;
#         border-radius: 8px;
#         padding: 1rem; /* Reduced padding */
#         margin-bottom: 0.8rem; /* Reduced margin */
#         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
#         border: 1px solid #e2e8f0;
#         height: 100%;
#     }
    
#     .card-header {
#         font-size: 1rem; /* Smaller font */
#         font-weight: 600;
#         color: #2d3748;
#         margin-bottom: 0.75rem; /* Reduced margin */
#         padding-bottom: 0.5rem; /* Reduced padding */
#         border-bottom: 1px solid #e2e8f0;
#     }
    
#     /* Video info */
#     .video-info {
#         display: flex;
#         gap: 0.8rem; /* Reduced gap */
#         margin-bottom: 0.8rem; /* Reduced margin */
#     }
    
#     .video-thumbnail {
#         flex-shrink: 0;
#         width: 140px; /* Slightly smaller thumbnail */
#         height: 78px; /* Adjusted height for aspect ratio */
#         border-radius: 6px;
#         overflow: hidden;
#     }
    
#     .video-thumbnail img {
#         width: 100%;
#         height: 100%;
#         object-fit: cover;
#     }
    
#     .video-details {
#         flex: 1;
#     }
    
#     .video-title {
#         font-size: 0.95rem; /* Slightly smaller font */
#         font-weight: 600;
#         color: #1a202c;
#         margin-bottom: 0.4rem; /* Reduced margin */
#         line-height: 1.3;
#     }
    
#     .video-meta {
#         font-size: 0.8rem; /* Smaller font */
#         color: #4a5568;
#         margin-bottom: 0.2rem; /* Reduced margin */
#     }
    
#     .video-stats {
#         display: flex;
#         gap: 0.8rem; /* Reduced gap */
#         font-size: 0.8rem; /* Smaller font */
#         color: #4a5568;
#     }
    
#     /* Sentiment grid */
#     .sentiment-grid {
#         display: grid;
#         grid-template-columns: repeat(3, 1fr);
#         gap: 0.8rem; /* Reduced gap */
#         margin-bottom: 0.8rem; /* Reduced margin */
#     }
    
#     .sentiment-item {
#         background: #f8fafc;
#         border-radius: 6px;
#         padding: 0.8rem; /* Reduced padding */
#         text-align: center;
#         border-left: 3px solid #e2e8f0;
#     }
    
#     .sentiment-item.positive { border-left-color: #38a169; }
#     .sentiment-item.negative { border-left-color: #e53e3e; }
#     .sentiment-item.neutral { border-left-color: #d69e2e; }
    
#     .sentiment-value {
#         font-size: 1.2rem; /* Smaller font */
#         font-weight: 700;
#         margin: 0.4rem 0; /* Reduced margin */
#         color: #4a5568;

#     }
    
#     .sentiment-label {
#         font-size: 0.8rem; /* Smaller font */
#         color: #4a5568;
#     }
    
#     /* Comments */
#     .comments-container {
#         padding-right: 0.5rem; 
#     }
    
#     .comment-section-header {
#         font-size: 0.85rem; /* Smaller font */
#         font-weight: 600;
#         color: white;
#         padding: 0.4rem; /* Reduced padding */
#         border-radius: 4px;
#         margin-bottom: 0.6rem; /* Reduced margin */
#         text-align: center;
#     }
    
#     .comment-section-header.positive {
#         background: #38a169;
#     }
    
#     .comment-section-header.negative {
#         background: #e53e3e;
#     }
#     .comment-section-header.neutral {
#         background: #d69e2e;
#     }
    
#     .comment-item {
#         background: #f8fafc;
#         border-radius: 6px;
#         padding: 0.8rem; /* Reduced padding */
#         margin-bottom: 0.5rem; /* Reduced margin */
#         border-left: 3px solid #e2e8f0;
#     }
    
#     .comment-text {
#         font-size: 0.85rem; /* Smaller font */
#         line-height: 1.4; /* Slightly reduced line height */
#         color: #2d3748;
#         margin-bottom: 0.4rem; /* Reduced margin */
#     }
    
#     .comment-meta {
#         font-size: 0.75rem; /* Smaller font */
#         color: #718096;
#         display: flex;
#         justify-content: space-between;
#     }
    
#     /* Empty state */
#     .empty-state {
#         text-align: center;
#         color: #718096;
#         font-size: 0.85rem; /* Smaller font */
#         padding: 1.5rem 0.8rem; /* Reduced padding */
#         background: #f8fafc;
#         border-radius: 6px;
#         border: 1px dashed #cbd5e0;
#     }
    
#     /* Hide Streamlit elements */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     .stDeployButton {visibility: hidden;}
    
#     /* Responsive adjustments */
#     @media (max-width: 768px) {
#         .header-title {
#             font-size: 1.8rem;
#         }
        
#         .video-info {
#             flex-direction: column;
#         }
        
#         .video-thumbnail {
#             width: 100%;
#             height: 160px;
#         }
        
#         .sentiment-grid {
#             grid-template-columns: 1fr;
#         }
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'comments_data' not in st.session_state:
#     st.session_state.comments_data = None
# if 'analysis_results' not in st.session_state:
#     st.session_state.analysis_results = None
# if 'video_info' not in st.session_state:
#     st.session_state.video_info = None
# if 'custom_text_result' not in st.session_state:
#     st.session_state.custom_text_result = None

# # Constants
# MAX_COMMENTS = 1000  # Maximum number of comments to fetch

# @st.cache_resource
# def load_models():
#     """Load trained models and preprocessing components for Smart Gating Ensemble"""
#     try:
#         # Load base components
#         tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
#         label_encoder = joblib.load('models/label_encoder.pkl')
        
#         # Load the complete ensemble package
#         ensemble_package = joblib.load('models/smart_gating_ensemble.pkl')
        
#         # Verify all required components exist
#         required_components = [
#             'svm_model', 'bilstm_model', 'gating_classifier',
#             'tokenizer', 'max_sequence_length'
#         ]
        
#         if not all(key in ensemble_package for key in required_components):
#             st.error("Missing components in ensemble package")
#             return None, None, None, None, None, None
            
#         return (
#             ensemble_package,  # model
#             tfidf_vectorizer,
#             label_encoder,
#             ensemble_package['tokenizer'],
#             {'best_model_type': 'smart_gating'},
#             ensemble_package['max_sequence_length']
#         )

#     except Exception as e:
#         st.error(f"Error loading models: {str(e)}")
#         st.stop()
#         return None, None, None, None, None, None

# def preprocess_text(text):
#     """Preprocess text for sentiment analysis"""
#     if pd.isna(text):
#         return ""
    
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     text = re.sub(r'@\w+|#\w+', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
#     text = re.sub(r'\d+', '', text)
#     return text.strip()

# def extract_video_id(url):
#     """Extract video ID from YouTube URL"""
#     patterns = [
#         r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
#         r'(?:embed\/)([0-9A-Za-z_-]{11})',
#         r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
#         r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
#     ]
    
#     for pattern in patterns:
#         match = re.search(pattern, url)
#         if match:
#             return match.group(1)
#     return None

# def get_video_info(video_id):
#     """Fetch video information from YouTube Data API"""
#     try:
#         url = f"{YOUTUBE_API_BASE_URL}/videos"
#         params = {
#             'part': 'snippet,statistics',
#             'id': video_id,
#             'key': YOUTUBE_API_KEY
#         }

#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         data = response.json()

#         if not data['items']:
#             return None

#         video = data['items'][0]
#         snippet = video['snippet']
#         statistics = video['statistics']

#         def format_number(num_str):
#             try:
#                 num = int(num_str)
#                 if num >= 1000000:
#                     return f"{num/1000000:.1f}M"
#                 elif num >= 1000:
#                     return f"{num/1000:.1f}K"
#                 else:
#                     return str(num)
#             except:
#                 return num_str

#         return {
#             'title': snippet['title'],
#             'description': snippet['description'][:150] + "..." if len(snippet['description']) > 150 else snippet['description'],
#             'channel_title': snippet['channelTitle'],
#             'published_at': snippet['publishedAt'][:10],
#             'thumbnail_url': snippet['thumbnails']['high']['url'],
#             'view_count': format_number(statistics.get('viewCount', '0')),
#             'like_count': format_number(statistics.get('likeCount', '0')),
#             'comment_count': format_number(statistics.get('commentCount', '0')),
#         }

#     except Exception as e:
#         st.error(f"Error fetching video info: {str(e)}")
#         return None

# def fetch_youtube_comments(video_id):
#     """Fetch ALL YouTube comments using YouTube Data API"""
#     try:
#         comments_data = []
#         next_page_token = None
#         total_fetched = 0
        
#         progress_bar = st.progress(0, text="Fetching comments...")
        
#         while True:
#             url = f"{YOUTUBE_API_BASE_URL}/commentThreads"
#             params = {
#                 'part': 'snippet',
#                 'videoId': video_id,
#                 'key': YOUTUBE_API_KEY,
#                 'maxResults': 100,  # Maximum allowed per request
#                 'order': 'relevance'
#             }

#             if next_page_token:
#                 params['pageToken'] = next_page_token

#             response = requests.get(url, params=params)
#             response.raise_for_status()
#             data = response.json()

#             if not data.get('items'):
#                 break

#             for item in data['items']:
#                 comment = item['snippet']['topLevelComment']['snippet']
#                 comments_data.append({
#                     'comment': comment['textDisplay'],
#                     'author': comment['authorDisplayName'],
#                     'published_at': comment['publishedAt'][:10],
#                     'like_count': comment['likeCount']
#                 })
#                 total_fetched += 1
                
#                 # Update progress bar
#                 progress_bar.progress(min(total_fetched/MAX_COMMENTS, 1.0), 
#                                     text=f"Fetched {total_fetched} comments")
                
#                 # Break if we've reached the max comments limit
#                 if total_fetched >= MAX_COMMENTS:
#                     break

#             # Break conditions
#             if total_fetched >= MAX_COMMENTS or not data.get('nextPageToken'):
#                 break
                
#             next_page_token = data.get('nextPageToken')
#             time.sleep(0.1)  # Be gentle with the API

#         progress_bar.empty()
#         return comments_data

#     except Exception as e:
#         st.error(f"Error fetching comments: {str(e)}")
#         return []

# def predict_sentiment(comments, model, tfidf_vectorizer, label_encoder, tokenizer, model_info, max_length):
#     """Predict sentiment using Smart Gating Ensemble"""
#     processed_comments = [preprocess_text(comment) for comment in comments]
#     sentiments = []
#     confidence_scores = []
#     gating_decisions = []  # Track when SVM overrides BiLSTM

#     if model_info.get('best_model_type') != 'smart_gating':
#         st.error("Incorrect model type loaded")
#         return ["Error"] * len(comments), [0.0] * len(comments)

#     # Unpack ensemble components
#     svm_model = model['svm_model']
#     bilstm_model = model['bilstm_model']
#     gate_clf = model['gating_classifier']
#     ensemble_tokenizer = model['tokenizer']
#     ensemble_max_length = model['max_sequence_length']

#     # 1. Get SVM predictions
#     svm_features = tfidf_vectorizer.transform(processed_comments)
#     svm_probs = svm_model.predict_proba(svm_features)
#     svm_preds = np.argmax(svm_probs, axis=1)

#     # 2. Get BiLSTM predictions
#     if not DEEP_LEARNING_AVAILABLE:
#         st.error("TensorFlow required for BiLSTM predictions")
#         return ["Error"] * len(comments), [0.0] * len(comments)
        
#     sequences = ensemble_tokenizer.texts_to_sequences(processed_comments)
#     padded_sequences = pad_sequences(sequences, maxlen=ensemble_max_length)
#     bilstm_probs = bilstm_model.predict(padded_sequences)
#     bilstm_preds = np.argmax(bilstm_probs, axis=1)

#     # 3. Create meta-features for gating
#     meta_features = np.hstack([svm_probs, bilstm_probs])

#     # 4. Make final predictions
#     for i in range(len(processed_comments)):
#         default_pred = bilstm_preds[i]  # Start with BiLSTM
        
#         # Check if models disagree
#         if svm_preds[i] != bilstm_preds[i]:
#             # Ask gating classifier which model to trust
#             gate_pred = gate_clf.predict(meta_features[i:i+1])[0]
            
#             if gate_pred == svm_preds[i]:  # Gate agrees with SVM
#                 final_pred = svm_preds[i]
#                 gating_decisions.append(True)  # SVM was used
#             else:
#                 final_pred = default_pred
#                 gating_decisions.append(False)
#         else:
#             final_pred = default_pred
#             gating_decisions.append(False)
        
#         # Get confidence from the chosen model
#         if final_pred == svm_preds[i]:
#             confidence = svm_probs[i][final_pred]
#         else:
#             confidence = bilstm_probs[i][final_pred]
            
#         sentiments.append(label_encoder.inverse_transform([final_pred])[0])
#         confidence_scores.append(confidence)

#     # Store gating stats in session state for visualization
#     st.session_state.gating_stats = {
#         'total': len(gating_decisions),
#         'svm_overrides': sum(gating_decisions),
#         'override_pct': (sum(gating_decisions) / len(gating_decisions)) * 100
#     }
    
#     return sentiments, confidence_scores

# def main():
#     # Load models
#     model, tfidf_vectorizer, label_encoder, tokenizer, model_info, max_length = load_models()

#     if model is None:
#         st.error("Could not load models. Please run the training notebook first.")
#         return

#     # Header at the very top
#     st.markdown("""
#     <div class="header-container">
#         <h1 class="header-title" style="margin: 0 !important; padding: 0 !important; color: #000000;text-align: center;font-size: 1.8rem;font-weight: 700;">YouTube Sentiment Analyzer</h1>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Main content container
#     st.markdown('<div class="main-content">', unsafe_allow_html=True)

#     # Input section - Simplified to just URL input
#     with st.container():
#         st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
#         video_url = st.text_input("YouTube Video URL", 
#                                 placeholder="https://www.youtube.com/watch?v=...",
#                                 key="youtube_url")
        
#         analyze_btn = st.button("Analyze Video", key="analyze_video", use_container_width=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)

#     # Main content columns
#     left_col, right_col = st.columns([1, 2])

#     # LEFT COLUMN - Video Info & Sentiment Summary
#     with left_col:
#         # Video Information Card
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown('<div class="card-header">Video Information</div>', unsafe_allow_html=True)
        
#         if st.session_state.video_info:
#             video_info = st.session_state.video_info
#             st.markdown(f"""
#             <div class="video-info">
#                 <div class="video-thumbnail">
#                     <img src="{video_info['thumbnail_url']}" alt="Video thumbnail">
#                 </div>
#                 <div class="video-details">
#                     <div class="video-title">{video_info['title']}</div>
#                     <div class="video-meta">{video_info['channel_title']}</div>
#                     <div class="video-meta">{video_info['published_at']}</div>
#                     <div class="video-stats">
#                         <span>👀 {video_info['view_count']}</span>
#                         <span>👍 {video_info['like_count']}</span>
#                         <span>💬 {video_info['comment_count']}</span>
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown("""
#             <div class="empty-state">
#                 Enter a YouTube URL to see video details
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)

#         # Sentiment Analysis Results
#         if st.session_state.analysis_results is not None:
#             st.markdown('<div class="card">', unsafe_allow_html=True)
#             st.markdown('<div class="card-header">Sentiment Analysis</div>', unsafe_allow_html=True)
            
#             results_df = st.session_state.analysis_results
#             total_comments = len(results_df)
            
#             # Show comment count info
#             st.markdown(f"""
#             <div style="margin-bottom: 1rem; padding: 0.5rem; background: #f0f4f8; border-radius: 6px;">
#                 <div style="font-size: 0.85rem; color: #4a5568; text-align: center;">
#                     Analyzed {total_comments} comments
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
            
#             if 'gating_stats' in st.session_state:
#                 gs = st.session_state.gating_stats
#                 st.markdown(f"""
#                 <div style="margin-bottom: 1rem; padding: 0.5rem; background: #f0f4f8; border-radius: 6px;">
#                     <div style="font-size: 0.85rem; color: #4a5568; text-align: center;">
#                         <b>Model Collaboration:</b> 
#                         SVM corrected BiLSTM in {gs['svm_overrides']} cases
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             if total_comments > 0:
#                 positive_count = len(results_df[results_df['sentiment'] == 'Positive'])
#                 negative_count = len(results_df[results_df['sentiment'] == 'Negative'])
#                 neutral_count = len(results_df[results_df['sentiment'] == 'Neutral'])

#                 positive_pct = (positive_count / total_comments) * 100
#                 negative_pct = (negative_count / total_comments) * 100
#                 neutral_pct = (neutral_count / total_comments) * 100

#                 st.markdown(f"""
#                 <div class="sentiment-grid">
#                     <div class="sentiment-item positive">
#                         <div class="sentiment-value">{positive_pct:.1f}%</div>
#                         <div class="sentiment-label">Positive</div>
#                     </div>
#                     <div class="sentiment-item neutral">
#                         <div class="sentiment-value">{neutral_pct:.1f}%</div>
#                         <div class="sentiment-label">Neutral</div>
#                     </div>
#                     <div class="sentiment-item negative">
#                         <div class="sentiment-value">{negative_pct:.1f}%</div>
#                         <div class="sentiment-label">Negative</div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             st.markdown('</div>', unsafe_allow_html=True)

#     # RIGHT COLUMN - Comments
#     with right_col:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown('<div class="card-header">Top Comments</div>', unsafe_allow_html=True)
        
#         if st.session_state.analysis_results is not None:
#             results_df = st.session_state.analysis_results
            
#             # Three columns for different sentiments
#             comment_col1, comment_col2, comment_col3 = st.columns(3)
#             sentiment_types = ['Positive', 'Negative', 'Neutral']
#             columns = [comment_col1, comment_col2, comment_col3]
            
#             for i, sentiment_type in enumerate(sentiment_types):
#                 with columns[i]:
#                     sentiment_df = results_df[results_df['sentiment'] == sentiment_type]
                    
#                     st.markdown(f"""
#                     <div class="comment-section-header {sentiment_type.lower()}">
#                         {sentiment_type} ({len(sentiment_df)})
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Show top 5 comments for each sentiment
#                     for _, row in sentiment_df.head(5).iterrows():
#                         sentiment_class = row['sentiment'].lower()
#                         comment_text = row['comment'][:120] + ('...' if len(row['comment']) > 120 else '')
#                         author_name = row['author'][:20] + ('...' if len(row['author']) > 20 else '')
                        
#                         st.markdown(f"""
#                         <div class="comment-item {sentiment_class}">
#                             <div class="comment-text">"{comment_text}"</div>
#                             <div class="comment-meta">
#                                 <span>👤 {author_name}</span>
#                                 <span>👍 {row['like_count']}</span>
#                             </div>
#                         </div>
#                         """, unsafe_allow_html=True)
#                     if len(sentiment_df) == 0:
#                         st.markdown(f'<div class="empty-state">No {sentiment_type.lower()} comments</div>', 
#                                    unsafe_allow_html=True)
#         else:
#             st.markdown("""
#             <div class="empty-state">
#                 Analyze a YouTube video to see comments
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)  # Close main-content

#     # Analysis Logic
#     if analyze_btn and video_url:
#         video_id = extract_video_id(video_url)
#         if not video_id:
#             st.error("Invalid YouTube URL format")
#             return

#         with st.spinner("Analyzing video... (this may take a while for videos with many comments)"):
#             # Get video info
#             video_info = get_video_info(video_id)
#             if not video_info:
#                 st.error("Could not fetch video information")
#                 return
            
#             st.session_state.video_info = video_info

#             # Get comments - now fetches all available comments
#             comments_data = fetch_youtube_comments(video_id)
#             if not comments_data:
#                 st.error("Could not fetch comments")
#                 return
            
#             st.info(f"Fetched {len(comments_data)} comments for analysis")
#             st.session_state.comments_data = comments_data

#             # Analyze sentiment
#             comments = [item['comment'] for item in comments_data]
#             sentiments, confidence_scores = predict_sentiment(
#                 comments, model, tfidf_vectorizer, label_encoder,
#                 tokenizer, model_info, max_length
#             )

#             # Store results
#             results_df = pd.DataFrame({
#                 'comment': comments,
#                 'sentiment': sentiments,
#                 'confidence': confidence_scores if confidence_scores is not None else [0.0] * len(comments),
#                 'author': [item['author'] for item in comments_data],
#                 'like_count': [item['like_count'] for item in comments_data],
#                 'published_at': [item['published_at'] for item in comments_data]
#             })
            
#             st.session_state.analysis_results = results_df
#             st.session_state.custom_text_result = None
#             st.rerun()

# if __name__ == "__main__":
#     main()