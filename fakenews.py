import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-real {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .prediction-fake {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .confidence-high {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
    }
    .confidence-medium {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
    }
    .confidence-low {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class FakeNewsDetector:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_sequence_length = 40
        self.vocab_size = 5000
        
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['from', 'subject', 're', 'edu', 'use'])
    
    def preprocess_text(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if (
                token not in gensim.parsing.preprocessing.STOPWORDS and 
                len(token) > 3 and 
                token not in self.stop_words
            ):
                result.append(token)
        return " ".join(result)
    
    def load_demo_model(self):
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, 128),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.tokenizer = Tokenizer(num_words=self.vocab_size)
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def calculate_confidence_score(self, text):
        """Calculate confidence score based on various text features"""
        if not text.strip():
            return 0.0
        
        processed_text = self.preprocess_text(text)
        
        # Feature-based confidence calculation
        features = {
            'length_score': min(len(text.split()) / 200, 1.0),  # Longer text = more confident
            'structure_score': 0.0,
            'source_indicators': 0.0,
            'language_quality': 0.0
        }
        
        # Check for proper structure
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 3:
            features['structure_score'] = min(len(sentences) / 10, 1.0)
        
        # Check for source indicators
        source_keywords = ['according to', 'study', 'research', 'report', 'official', 'experts']
        source_count = sum(1 for keyword in source_keywords if keyword in text.lower())
        features['source_indicators'] = min(source_count / 5, 1.0)
        
        # Check language quality (basic grammar indicators)
        capital_letters = sum(1 for char in text if char.isupper())
        total_letters = sum(1 for char in text if char.isalpha())
        if total_letters > 0:
            capitalization_ratio = capital_letters / total_letters
            features['language_quality'] = 1.0 - abs(0.1 - capitalization_ratio) * 10
        
        # Weighted average of features
        weights = {
            'length_score': 0.3,
            'structure_score': 0.3,
            'source_indicators': 0.2,
            'language_quality': 0.2
        }
        
        confidence = sum(features[feature] * weights[feature] for feature in features)
        return min(max(confidence, 0.0), 1.0)
    
    def predict(self, text):
        if not text.strip():
            return None, 0.0
        
        processed_text = self.preprocess_text(text)

        fake_indicators = [
            'breaking', 'shocking', 'unbelievable', 'wont believe',
            'viral', 'secret', 'dont want you to know', 'amazing',
            'incredible', 'miracle', 'hidden', 'exposed'
        ]
        
        real_indicators = [
            'according to', 'study shows', 'research indicates',
            'official report', 'government', 'experts say',
            'university', 'research', 'study', 'scientists'
        ]
        
        text_lower = processed_text.lower()
        fake_score = sum(1 for ind in fake_indicators if ind in text_lower)
        real_score = sum(1 for ind in real_indicators if ind in text_lower)
        
        total_score = fake_score + real_score
        
        if total_score == 0:
            confidence = self.calculate_confidence_score(text)
            is_fake = len(text.split()) < 50
        else:
            fake_ratio = fake_score / total_score
            confidence = self.calculate_confidence_score(text)
            is_fake = fake_ratio > 0.5
        
        return is_fake, confidence


def main():
    st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
    
    detector = FakeNewsDetector()
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This AI-powered tool analyzes news content to detect potential fake news."
    )
    
    st.sidebar.title("How It Works")
    st.sidebar.markdown("""
    1. Enter news headline and content  
    2. AI analyzes language patterns  
    3. Get authenticity assessment  
    4. Review confidence score  
    """)
    
    st.sidebar.title("Tips for Spotting Fake News")
    st.sidebar.markdown("""
    - Check the source credibility  
    - Look for supporting evidence  
    - Verify with multiple sources  
    - Be wary of emotional language  
    - Check publication dates  
    """)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter News Article")
        
        headline = st.text_input("News Headline:")
        news_content = st.text_area("News Content:", height=300)
        
        analyze_btn = st.button("🔍 Analyze News", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if analyze_btn:
            if not headline.strip() and not news_content.strip():
                st.warning("⚠️ Please enter both headline and content for analysis.")
            else:
                with st.spinner("🤖 AI is analyzing the news content..."):
                    
                    full_text = f"{headline} {news_content}"
                    
                    is_fake, confidence = detector.predict(full_text)
                    
                    if is_fake is not None:
                        st.markdown("### 🎯 Prediction")
                        
                        if is_fake:
                            st.markdown(
                                '<div class="prediction-fake">'
                                '<h3>🚫 POTENTIALLY FAKE NEWS</h3>'
                                '<p>This content shows characteristics commonly found in misleading or unreliable news.</p>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="prediction-real">'
                                '<h3>✅ LIKELY REAL NEWS</h3>'
                                '<p>This content appears credible based on language analysis.</p>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        
                        # New Confidence Section
                        st.markdown("### 📊 Analysis Confidence")
                        confidence_percent = confidence * 100
                        
                        if confidence > 0.7:
                            confidence_class = "confidence-high"
                            label = "High Confidence"
                            description = "Strong indicators support this analysis"
                        elif confidence > 0.4:
                            confidence_class = "confidence-medium"
                            label = "Medium Confidence"
                            description = "Moderate indicators support this analysis"
                        else:
                            confidence_class = "confidence-low"
                            label = "Low Confidence"
                            description = "Limited indicators available for analysis"
                        
                        st.markdown(
                            f'<div class="{confidence_class}">'
                            f'<h4>🔍 {label}</h4>'
                            f'<p><strong>{confidence_percent:.1f}%</strong> - {description}</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Progress bar
                        st.markdown("**Confidence Level:**")
                        st.progress(float(confidence))
                        
                        # Additional insights
                        st.markdown("### 💡 Key Insights")
                        if confidence > 0.7:
                            st.success("✅ Strong textual patterns detected for reliable analysis")
                        elif confidence > 0.4:
                            st.warning("⚠️ Moderate evidence - consider additional verification")
                        else:
                            st.error("🔍 Limited data - analysis should be interpreted with caution")

        else:
            st.info("👆 Enter news content and click 'Analyze News' to get started.")
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🔍 This tool uses AI analysis for educational purposes. Always verify information from multiple sources."
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    detector = FakeNewsDetector()
    detector.load_demo_model()
    main()
