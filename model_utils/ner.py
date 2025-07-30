import torch
import numpy as np
import pickle
import os
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# BiLSTM_NER model class exactly as in training
class BiLSTM_NER(torch.nn.Module):
    def __init__(self, vocab_size, pos_size, tag_size, emb_dim=100, pos_emb_dim=16, lstm_units=128):
        super().__init__()
        self.word_emb = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(pos_size, pos_emb_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(emb_dim + pos_emb_dim, lstm_units, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(lstm_units * 2, tag_size)

    def forward(self, Xw, Xp):
        w = self.word_emb(Xw)
        p = self.pos_emb(Xp)
        feats = torch.cat([w, p], dim=-1)
        lstm_out, _ = self.lstm(feats)
        out = self.fc(lstm_out)
        return out


def load_ner_resources():
    DATA_DIR = './data/ner'
    MODEL_DIR = './models/ner'

    with open(os.path.join(DATA_DIR, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'pos2idx.pkl'), 'rb') as f:
        pos2idx = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'tag2idx.pkl'), 'rb') as f:
        tag2idx = pickle.load(f)

    idx2tag = {i: t for t, i in tag2idx.items()}
    idx2word = {i: w for w, i in word2idx.items()}

    vocab_size = len(word2idx)
    pos_size = len(pos2idx)
    tag_size = len(tag2idx)

    config = dict(
        emb_dim=100,
        pos_emb_dim=16,
        lstm_units=128,
        max_len=35
    )

    model = BiLSTM_NER(vocab_size, pos_size, tag_size,
                       config['emb_dim'], config['pos_emb_dim'], config['lstm_units'])
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_ner_model.pt'), map_location=device))
    model.eval()
    model.to(device)

    vocabs = dict(
        word2idx=word2idx,
        pos2idx=pos2idx,
        tag2idx=tag2idx,
        idx2tag=idx2tag,
        idx2word=idx2word
    )
    return vocabs, model, config


def simple_pos(tokens):
    # Dummy POS tag indices; replace with real POS tagging if possible
    return [1] * len(tokens)


def tokenize(text):
    # Regex tokenizer: splits words and separates punctuation
    # For example, "Silas," -> ["Silas", ","]
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text, re.UNICODE)
    return tokens


def preprocess_input(text, word2idx, pos2idx, max_len):
    tokens = tokenize(text)
    # Map tokens to indices, use <UNK> idx=1 if missing
    token_idxs = [word2idx.get(t, word2idx.get("<UNK>", 1)) for t in tokens]
    pos_idxs = simple_pos(tokens)
    if len(token_idxs) > max_len:
        token_idxs = token_idxs[:max_len]
        pos_idxs = pos_idxs[:max_len]
        tokens = tokens[:max_len]
    else:
        pad_len = max_len - len(token_idxs)
        token_idxs += [0] * pad_len
        pos_idxs += [0] * pad_len
    return np.array(token_idxs), np.array(pos_idxs), tokens


def predict_ner(text, vocabs, model, config):
    if not text.strip():
        return []
    word2idx, pos2idx, idx2tag, idx2word = vocabs['word2idx'], vocabs['pos2idx'], vocabs['idx2tag'], vocabs['idx2word']
    max_len = config.get("max_len", 35)

    Xw, Xp, tokens = preprocess_input(text, word2idx, pos2idx, max_len)

    # Convert to batch size 1, tensor with proper device
    Xw = torch.LongTensor(np.expand_dims(Xw, axis=0)).to(device)
    Xp = torch.LongTensor(np.expand_dims(Xp, axis=0)).to(device)

    with torch.no_grad():
        logits = model(Xw, Xp)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    results = []
    for token, tag_idx in zip(tokens, preds[:len(tokens)]):
        tag = idx2tag.get(tag_idx, "O")
        results.append((token, tag))
    return results
