import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from torchtext.data.utils import get_tokenizer
from PyPDF2 import PdfReader

DEVICE = 'cuda'

# Sentiment Model
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

MAX_LEN = 100

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


# Summarization Model
class TransformerSummarizer(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, ff_dim, num_layers,
                 max_article_len, max_summary_len, pad_idx):
        super(TransformerSummarizer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_encoder = nn.Embedding(max_article_len, emb_dim)
        self.pos_decoder = nn.Embedding(max_summary_len, emb_dim)

        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )

        self.fc_out = nn.Linear(emb_dim, vocab_size)

        self.max_article_len = max_article_len
        self.max_summary_len = max_summary_len

    def forward(self, src, tgt):
        src_mask = None
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        src_pos = self.pos_encoder(torch.arange(self.max_article_len, device=src.device)).unsqueeze(0)
        tgt_pos = self.pos_decoder(torch.arange(self.max_summary_len, device=tgt.device)).unsqueeze(0)

        src_emb = self.embedding(src) + src_pos[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + tgt_pos[:, :tgt.size(1), :]

        outs = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(outs)


# Hyperparameters
MAX_ARTICLE_LEN = 400
MAX_SUMMARY_LEN = 50

@st.cache_resource
def load_summarization_resources():
    vocab_path = './models/summarization/vocab.json'
    model_path = './models/summarization/best_summarization_model.pt'

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    pad_idx = vocab.get('<PAD>', 0)

    inv_vocab = {int(v): k for k, v in vocab.items()}

    model = TransformerSummarizer(
        vocab_size=len(vocab),
        emb_dim=256,
        nhead=8,
        ff_dim=512,
        num_layers=4,
        max_article_len=MAX_ARTICLE_LEN,
        max_summary_len=MAX_SUMMARY_LEN,
        pad_idx=pad_idx
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return vocab, inv_vocab, model


def preprocess_text_summarization(text, vocab, max_len=MAX_ARTICLE_LEN):
    tokens = text.lower().split()[:max_len]
    unk_idx = vocab.get('', 1)
    pad_idx = vocab.get('', 0)
    token_ids = [vocab.get(t, unk_idx) for t in tokens]
    if len(token_ids) < max_len:
        token_ids += [pad_idx] * (max_len - len(token_ids))
    return token_ids


def beam_search_decode_transformer(model, src_indices, vocab, idx2word,
                                   beam_width=4,
                                   max_summary_len=MAX_SUMMARY_LEN,
                                   alpha=0.7):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        src = torch.tensor([src_indices], dtype=torch.long, device=device)

        src_pos = model.pos_encoder(torch.arange(model.max_article_len, device=device)).unsqueeze(0)
        src_emb = model.embedding(src) + src_pos[:, :src.size(1), :]

        memory = model.transformer.encoder(src_emb)

        start_id = vocab.get('', 2)  # start token index 2
        end_id = vocab.get('', 3)    # end token index 3

        beams = [(0.0, [start_id])]

        for _ in range(max_summary_len):
            all_candidates = []
            for score, seq in beams:
                if seq[-1] == end_id:
                    all_candidates.append((score, seq))
                    continue

                tgt_seq = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_pos = model.pos_decoder(torch.arange(len(seq), device=device)).unsqueeze(0)
                tgt_emb = model.embedding(tgt_seq) + tgt_pos
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)

                decoder_output = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = model.fc_out(decoder_output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                top_log_probs, top_ids = torch.topk(log_probs, beam_width)

                for log_p, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                    new_score = score + log_p
                    new_seq = seq + [token_id]
                    all_candidates.append((new_score, new_seq))

            # Length normalization
            beams = sorted(all_candidates, key=lambda x: x[0]/(len(x[1]) ** alpha), reverse=True)[:beam_width]

            if all(seq[-1] == end_id for _, seq in beams):
                break

        best_seq = beams[0][1]

        # Remove start token and everything after end token
        if end_id in best_seq:
            best_seq = best_seq[1:best_seq.index(end_id)]
        else:
            best_seq = best_seq[1:]

        summary = ' '.join(idx2word.get(i, '') for i in best_seq)
        return summary


def generate_summary(text, vocab, inv_vocab, model):
    src_indices = preprocess_text_summarization(text, vocab)
    summary = beam_search_decode_transformer(model, src_indices, vocab, inv_vocab)
    return summary


# Utilities
def extract_text_from_pdf(uploaded_file):
    text = ''
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + '\n'
    except:
        text = ''
    return text


# Streamlit UI
def main():
    st.title('AI Powered Research Assistant')

    uploaded_file = st.file_uploader('Upload a PDF file', type=['pdf'])
    input_text = ''

    if uploaded_file is not None:
        extracted = extract_text_from_pdf(uploaded_file)
        st.text_area('Extracted Text from PDF', extracted, height=150, disabled=True)
        input_text = extracted
    else:
        input_text = st.text_area('Or paste your text here', '', height=200)

    feature = st.selectbox('Select a feature to analyze:', ['Summarization', 'Q&A', 'NER', 'Sentiment Analysis'])

    question = ''
    if feature == 'Q&A':
        question = st.text_input('Enter your question related to the context')

    if st.button('Analyze'):
        if not input_text.strip():
            st.warning('Please provide some input text or upload a PDF.')
            return

        if feature == 'Sentiment Analysis':
            vocab_sent, model_sent = load_sentiment_resources()
            sentiment, conf = predict_sentiment(input_text, vocab_sent, model_sent)
            st.success(f'Sentiment: {sentiment} (Confidence: {conf:.2f})')

        elif feature == 'Summarization':
            vocab_sum, inv_vocab_sum, model_sum = load_summarization_resources()
            summary = generate_summary(input_text, vocab_sum, inv_vocab_sum, model_sum)
            st.subheader('Summary:')
            st.write(summary)

        elif feature == 'Q&A':
            st.info('Q&A model integration not implemented yet.')
            if question.strip():
                vocab_sent, model_sent = load_sentiment_resources()
                sentiment, conf = predict_sentiment(question, vocab_sent, model_sent)
                st.info(f'Question Sentiment: {sentiment} (Confidence: {conf:.2f})')
            else:
                st.warning('Please enter a question for Q&A feature.')

        elif feature == 'NER':
            st.info('NER model integration not implemented yet.')

if __name__ == '__main__':
    main()
