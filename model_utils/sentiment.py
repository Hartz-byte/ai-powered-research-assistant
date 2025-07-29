import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
import streamlit as st

DEVICE = 'cuda'
MAX_LEN = 100
    
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim, n_layers=1, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            dropout=dropout if n_layers > 1 else 0,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dropped = self.dropout(hidden_cat)
        output = self.fc(dropped)
        return output

@st.cache_resource
def load_sentiment_resources():
    with open('./models/sentiment/vocab.json', 'r') as f:
        vocab = json.load(f)
    model = BiLSTMClassifier(embedding_dim=64, hidden_dim=120, vocab_size=len(vocab), output_dim=3, n_layers=1, dropout=0.5)
    model.load_state_dict(torch.load('./models/sentiment/best_sentiment_model.pt', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return vocab, model

def preprocess_text_sentiment(text, vocab):
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(text.lower())
    indices = [vocab.get(tok, 0) for tok in tokens]
    if len(indices) < MAX_LEN:
        indices += [0] * (MAX_LEN - len(indices))
    else:
        indices = indices[:MAX_LEN]
    return torch.tensor([indices], dtype=torch.long)

def predict_sentiment(text, vocab, model):
    input_tensor = preprocess_text_sentiment(text, vocab).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    labels = ['Negative', 'Neutral', 'Positive']
    return labels[pred.item()], confidence.item()
