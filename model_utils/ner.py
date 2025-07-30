import os
import torch
import torch.nn as nn
import pickle
import spacy
from typing import List, Tuple

# Load SpaCy small English model for tokenization and POS tagging
# Make sure you have run: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DIR = './models/ner/best_ner_model.pt'

# Load vocabs dicts and model from checkpoint
def load_ner_resources():
    checkpoint = torch.load(MODEL_DIR, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']
    vocabs = checkpoint['vocabularies']

    # Add pad_idx to config (fallback to 0 if '' not found)
    pad_idx = vocabs['word2idx'].get('', 0)
    config['pad_idx'] = pad_idx

    # Define BiLSTM-CRF model class and CRF inside this file (simple version)
    # -- CRF implementation (from your training code) --

    class CRF(nn.Module):
        def __init__(self, num_tags, batch_first=True):
            super().__init__()
            self.num_tags = num_tags
            self.batch_first = batch_first
            self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
            self.transitions.data[pad_idx, :] = -10000  # no transitions from PAD
            self.transitions.data[:, pad_idx] = -10000  # no transitions to PAD

        def _compute_partition_function(self, emissions, mask):
            batch_size, seq_length, num_tags = emissions.size()
            forward_var = emissions[:, 0]
            for i in range(1, seq_length):
                emit_score = emissions[:, i].unsqueeze(1)
                trans_score = self.transitions.unsqueeze(0)
                next_tag_var = forward_var.unsqueeze(2) + trans_score + emit_score
                next_tag_var = torch.logsumexp(next_tag_var, dim=1)
                forward_var = torch.where(mask[:, i].unsqueeze(1), next_tag_var, forward_var)
            terminal_var = torch.logsumexp(forward_var, dim=1)
            return terminal_var

        def _compute_score(self, emissions, tags, mask):
            batch_size, seq_length = tags.size()
            emission_scores = torch.gather(emissions, 2, tags.unsqueeze(2)).squeeze(2)
            emission_scores = emission_scores * mask.float()
            emission_scores = emission_scores.sum(dim=1)
            transition_scores = torch.zeros(batch_size, device=emissions.device)
            for i in range(seq_length - 1):
                curr_tags = tags[:, i]
                next_tags = tags[:, i + 1]
                valid_mask = mask[:, i + 1]
                transition_scores += self.transitions[curr_tags, next_tags] * valid_mask.float()
            return emission_scores + transition_scores

        def forward(self, emissions, tags, mask=None):
            if mask is None:
                mask = torch.ones_like(tags, dtype=torch.bool)
            partition = self._compute_partition_function(emissions, mask)
            sequence_score = self._compute_score(emissions, tags, mask)
            return (partition - sequence_score).mean()

        def decode(self, emissions, mask=None):
            if mask is None:
                mask = torch.ones(emissions.size()[:2], dtype=torch.bool, device=emissions.device)
            batch_size, seq_length, num_tags = emissions.size()
            viterbi_vars = emissions[:, 0]
            backpointers = []
            for i in range(1, seq_length):
                broadcast_vars = viterbi_vars.unsqueeze(2)  # (batch, tags, 1)
                broadcast_trans = self.transitions.unsqueeze(0)  # (1, tags, tags)
                next_tag_vars = broadcast_vars + broadcast_trans
                best_tag_scores, best_tags = torch.max(next_tag_vars, dim=1)
                backpointers.append(best_tags)
                viterbi_vars = best_tag_scores + emissions[:, i]
                viterbi_vars = torch.where(mask[:, i].unsqueeze(1), viterbi_vars, viterbi_vars)
            best_paths = []
            for b in range(batch_size):
                seq_len = mask[b].sum().item()
                best_last_tag = torch.argmax(viterbi_vars[b]).item()
                best_path = [best_last_tag]
                for backptrs_t in reversed(backpointers[:seq_len-1]):
                    best_last_tag = backptrs_t[b][best_last_tag].item()
                    best_path.append(best_last_tag)
                best_path.reverse()
                best_paths.append(best_path)
            return best_paths

    class BiLSTM_CRF_Enhanced(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.pad_idx = config['pad_idx']
            self.word_embedding = nn.Embedding(config['vocab_size'], config['word_embed_dim'], padding_idx=self.pad_idx)
            self.pos_embedding = nn.Embedding(config['pos_vocab_size'], config['pos_embed_dim'])
            total_embed_dim = config['word_embed_dim'] + config['pos_embed_dim'] + config['char_embed_dim']
            self.lstm = nn.LSTM(total_embed_dim, config['hidden_dim'], num_layers=config['num_layers'],
                                batch_first=True, bidirectional=True,
                                dropout=config['dropout'] if config['num_layers'] > 1 else 0)
            self.layer_norm = nn.LayerNorm(config['hidden_dim'] * 2)
            self.dropout = nn.Dropout(config['dropout'])
            self.hidden2tag = nn.Linear(config['hidden_dim'] * 2, config['tag_vocab_size'])
            self.crf = CRF(config['tag_vocab_size'], batch_first=True)

            nn.init.uniform_(self.word_embedding.weight, -0.1, 0.1)
            nn.init.uniform_(self.pos_embedding.weight, -0.1, 0.1)
            nn.init.xavier_uniform_(self.hidden2tag.weight)
            nn.init.constant_(self.hidden2tag.bias, 0)

            self.char_embed_dim = config['char_embed_dim']
            # self.char_cnn = self._build_char_cnn()
            self.char_cnn = nn.Sequential(
                nn.Conv1d(self.char_embed_dim, 50, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.25)
            )
        
        def _build_char_cnn(self):
            return nn.Sequential(
                nn.Conv1d(self.char_embed_dim, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Dropout(0.25)
            )

        def _get_char_features(self, words, idx2word):
            batch_size, seq_len = words.size()
            device = words.device
            char_features = torch.zeros(batch_size, seq_len, self.char_embed_dim, device=device)
            for i in range(batch_size):
                for j in range(seq_len):
                    word_idx = words[i, j].item()
                    if word_idx != self.pad_idx:
                        word = idx2word.get(word_idx, '')
                        for k, char in enumerate(word[:self.char_embed_dim]):
                            char_features[i, j, k] = ord(char) % 128
            return char_features

        def forward(self, words, pos, tags=None, idx2word=None):
            mask = (words != self.pad_idx)
            word_embeds = self.word_embedding(words)
            pos_embeds = self.pos_embedding(pos)
            char_features = self._get_char_features(words, idx2word) if idx2word else torch.zeros_like(word_embeds[:, :, :self.char_embed_dim])
            embeds = torch.cat([word_embeds, pos_embeds, char_features], dim=2)
            embeds = self.dropout(embeds)
            lstm_out, _ = self.lstm(embeds)
            lstm_out = self.layer_norm(lstm_out)
            lstm_out = self.dropout(lstm_out)
            emissions = self.hidden2tag(lstm_out)
            if tags is not None:
                return self.crf(emissions, tags, mask)
            else:
                return self.crf.decode(emissions, mask)

    model = BiLSTM_CRF_Enhanced(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return vocabs, model, config


def preprocess_text_ner(text: str, word2idx: dict, pos2idx: dict, max_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    doc = nlp(text)
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]

    UNK_IDX = word2idx.get('', 0)
    word_ids = [word2idx.get(tok.lower(), UNK_IDX) for tok in tokens]
    pos_ids = [pos2idx.get(pos, 0) for pos in pos_tags]

    # Pad or truncate
    if len(word_ids) > max_seq_len:
        word_ids = word_ids[:max_seq_len]
        pos_ids = pos_ids[:max_seq_len]
        tokens = tokens[:max_seq_len]
    else:
        pad_len = max_seq_len - len(word_ids)
        word_ids.extend([UNK_IDX] * pad_len)
        pos_ids.extend([0] * pad_len)
        tokens.extend([''] * pad_len)

    words_tensor = torch.LongTensor([word_ids]).to(DEVICE)
    pos_tensor = torch.LongTensor([pos_ids]).to(DEVICE)

    return words_tensor, pos_tensor, tokens


def predict_ner(text: str, vocabs: dict, model: nn.Module, config: dict) -> List[Tuple[str, str]]:
    word2idx = vocabs['word2idx']
    pos2idx = vocabs['pos2idx']
    idx2tag = vocabs['idx2tag']
    idx2word = vocabs['idx2word']
    max_seq_len = config.get('max_seq_len', 35)  # default max_len if missing

    words_tensor, pos_tensor, tokens = preprocess_text_ner(text, word2idx, pos2idx, max_seq_len)
    # Run model forward to get predicted tag sequences
    with torch.no_grad():
        predicted_tag_idxs = model(words_tensor, pos_tensor, idx2word=idx2word)[0]

    # Align tokens and tags (ignore padding tokens at end)
    ner_results = []
    for token, tag_idx in zip(tokens, predicted_tag_idxs):
        tag_label = idx2tag.get(tag_idx, 'O')
        if token.strip() != '':
            ner_results.append((token, tag_label))

    return ner_results
