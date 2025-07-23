import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
import os

# === Configuration ===
MODEL_PATH = './models/sentiment/best_sentiment_model.pt'
VOCAB_PATH = './models/sentiment/vocab.json'
MAX_LEN = 100
EMBEDDING_DIM = 64
HIDDEN_DIM = 120
NUM_LAYERS = 1
DROPOUT = 0.5

# === Model Definition ===
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                 output_dim=3, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(hidden_cat)
        return self.fc(out)

# === Utility Functions ===
def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r"[^a-z0-9\s']", '', text)
    tokens = text.split()
    return tokens

def preprocess(text, vocab, max_len=MAX_LEN):
    tokens = clean_and_tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(ids) < max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor([ids], dtype=torch.long)

def predict(text, model, vocab, device):
    model.eval()
    x = preprocess(text, vocab).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        class_id = probs.argmax(dim=1).item()
        confidence = float(probs[0, class_id])
        class_map_reverse = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return class_map_reverse[class_id], confidence

# === Load Model and Vocab ===
@st.cache_resource
def load_resources():
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    model = BiLSTMClassifier(vocab_size)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    return model, vocab, device

model, vocab, device = load_resources()

# === Streamlit UI ===
st.title("Sentiment Detection Demo")
st.write("Enter a sentence to detect its sentiment (Negative, Neutral, Positive).")

text = st.text_area("Your text:", "", height=100)

if st.button("Analyze Sentiment"):
    if text.strip():
        sentiment, confidence = predict(text, model, vocab, device)
        st.success(f"**Sentiment:** {sentiment}  (confidence: {confidence:.2f})")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.write("**About:** This demo uses a custom-trained BiLSTM sentiment model built from scratch without pre-trained components.")
