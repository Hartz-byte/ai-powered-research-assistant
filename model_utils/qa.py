import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
import string
import streamlit as st

DEVICE = 'cuda'

# Model classes from training notebook
class Highway(nn.Module):
    def __init__(self, size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = F.relu(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

class BiDAFModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=128, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.highway = Highway(embed_dim, num_layers=2)

        self.context_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0)
        self.question_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0)

        self.att_weight_c = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.att_weight_q = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.att_weight_cq = nn.Linear(2 * hidden_dim, 1, bias=False)

        self.modeling_lstm1 = nn.LSTM(8 * hidden_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout if dropout > 0 else 0)
        self.modeling_lstm2 = nn.LSTM(2 * hidden_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout if dropout > 0 else 0)

        self.start_linear = nn.Linear(10 * hidden_dim, 1)
        self.end_linear = nn.Linear(10 * hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, context, question):
        batch_size = context.size(0)
        context_len = context.size(1)
        question_len = question.size(1)

        context_mask = (context != self.embedding.padding_idx).float()
        question_mask = (question != self.embedding.padding_idx).float()

        context_emb = self.embedding(context)
        question_emb = self.embedding(question)

        context_emb = self.highway(context_emb)
        question_emb = self.highway(question_emb)

        context_emb = self.embedding_dropout(context_emb)
        question_emb = self.embedding_dropout(question_emb)

        context_enc, _ = self.context_lstm(context_emb)
        question_enc, _ = self.question_lstm(question_emb)

        similarity = self._compute_similarity(context_enc, question_enc)
        question_mask_expanded = question_mask.unsqueeze(1).expand(-1, context_len, -1)
        similarity = similarity.masked_fill(question_mask_expanded == 0, -1e9)

        c2q_att = F.softmax(similarity, dim=2)
        c2q = torch.bmm(c2q_att, question_enc)

        max_similarity = torch.max(similarity, dim=2)[0]
        q2c_att = F.softmax(max_similarity, dim=1)
        q2c = torch.bmm(q2c_att.unsqueeze(1), context_enc)
        q2c = q2c.expand(-1, context_len, -1)

        G = torch.cat([context_enc, c2q, context_enc * c2q, context_enc * q2c], dim=2)
        G = self.dropout(G)

        M1, _ = self.modeling_lstm1(G)
        M2, _ = self.modeling_lstm2(M1)

        start_input = torch.cat([G, M1], dim=2)
        end_input = torch.cat([G, M2], dim=2)

        start_logits = self.start_linear(start_input).squeeze(-1)
        end_logits = self.end_linear(end_input).squeeze(-1)

        start_logits = start_logits.masked_fill(context_mask == 0, -1e9)
        end_logits = end_logits.masked_fill(context_mask == 0, -1e9)

        return start_logits, end_logits

    def _compute_similarity(self, context_enc, question_enc):
        batch_size, context_len, hidden_size = context_enc.size()
        question_len = question_enc.size(1)

        context_expanded = context_enc.unsqueeze(2).expand(-1, -1, question_len, -1)
        question_expanded = question_enc.unsqueeze(1).expand(-1, context_len, -1, -1)
        elementwise_prod = context_expanded * question_expanded

        alpha = (self.att_weight_c(context_expanded) + 
                 self.att_weight_q(question_expanded) + 
                 self.att_weight_cq(elementwise_prod))

        return alpha.squeeze(-1)


# Load model and vocab cached 
@st.cache_resource
def load_qa_resources():
    model_path = './models/qa/best_qa_model.pt'
    # Load checkpoint dict
    checkpoint = torch.load(model_path, map_location=DEVICE)
    vocab_data = checkpoint['vocab']
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']
    config = checkpoint['config']

    # Create model with config params matching training
    model = BiDAFModel(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config.get('dropout', 0.2),
        pad_idx=word2idx.get('', 0)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return word2idx, idx2word, model


# Text processing utilities
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def simple_tokenize(text):
    text = re.sub(r"([.!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", text)
    return text.strip().split()

def encode_text(text, word2idx, max_len):
    tokens = simple_tokenize(clean_text(text.lower()))
    ids = [word2idx.get(token, word2idx.get('', 0)) for token in tokens[:max_len]]
    ids += [word2idx.get('', 0)] * (max_len - len(ids))
    return ids, tokens[:max_len]


# Inference function to get best answer span
def get_best_span(start_logits, end_logits, max_answer_len=30):
    start_probs = F.softmax(start_logits, dim=0)
    end_probs = F.softmax(end_logits, dim=0)
    best_score = 0
    best_start = 0
    best_end = 0
    for start_idx in range(len(start_probs)):
        for end_idx in range(start_idx, min(start_idx+max_answer_len, len(end_probs))):
            score = start_probs[start_idx] * end_probs[end_idx]
            if score > best_score:
                best_score = score
                best_start = start_idx
                best_end = end_idx
    return best_start, best_end, best_score

def predict_answer(context, question, word2idx, idx2word, model, max_context_len=400, max_question_len=50):
    context_ids, context_tokens = encode_text(context, word2idx, max_context_len)
    question_ids, question_tokens = encode_text(question, word2idx, max_question_len)

    context_tensor = torch.tensor([context_ids], dtype=torch.long).to(DEVICE)
    question_tensor = torch.tensor([question_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        start_logits, end_logits = model(context_tensor, question_tensor)

    start_idx, end_idx, confidence = get_best_span(start_logits[0], end_logits[0])
    if start_idx < len(context_tokens) and end_idx < len(context_tokens):
        answer_tokens = context_tokens[start_idx:end_idx + 1]
        answer_text = ' '.join(answer_tokens)
    else:
        answer_text = ""
        confidence = 0.0

    return {
        'answer': answer_text,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'confidence': confidence,
        'context_tokens': context_tokens,
        'question_tokens': question_tokens
    }
