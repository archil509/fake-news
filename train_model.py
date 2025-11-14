
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class FakeNewsModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_sequence_length = 500
        self.vocab_size = 5000
        
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        # Combine title and text
        df['content'] = df['title'] + ' ' + df['text']
        
        # Preprocess text
        stop_words = set(stopwords.words('english'))
        stop_words.update(['from', 'subject', 're', 'edu', 'use'])
        
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            return ' '.join(tokens)
        
        df['cleaned_content'] = df['content'].apply(clean_text)
        return df
    
    def prepare_sequences(self, texts):
        """Convert texts to sequences"""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        return padded_sequences
    
    def create_model(self):
        """Create the neural network model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 100, input_length=self.max_sequence_length),
            tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, df):
        """Train the model"""
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Prepare features and labels
        X = self.prepare_sequences(df['cleaned_content'].values)
        y = df['isfake'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        self.model = self.create_model()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model and tokenizer
        self.model.save('fake_news_model.h5')
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        return history

# Example usage
if __name__ == "__main__":
    # Load your dataset
    # df_true = pd.read_csv("True.csv")
    # df_fake = pd.read_csv("Fake.csv")
    # df = pd.concat([df_true, df_fake])
    
    print("Model training script ready!")
    print("Uncomment the dataset loading code to train with your data.")