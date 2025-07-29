import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

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

DEVICE = 'cuda'
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
