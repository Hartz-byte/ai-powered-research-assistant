import pickle
import os

def load_ner_vocabs():
    vocab_path = './data/ner/ner_vocabularies.pkl'
    with open(vocab_path, 'rb') as f:
        ner_vocabularies = pickle.load(f)
    
    word2idx = ner_vocabularies['word2idx']
    idx2word = ner_vocabularies['idx2word']
    pos2idx = ner_vocabularies['pos2idx']
    idx2pos = ner_vocabularies['idx2pos']
    tag2idx = ner_vocabularies['tag2idx']
    idx2tag = ner_vocabularies['idx2tag']
    return word2idx, idx2word, pos2idx, idx2pos, tag2idx, idx2tag

def create_char_vocab_from_words(word2idx):
    chars = set()
    for word in word2idx.keys():
        for c in word:
            chars.add(c)
    # Special tokens
    chars = sorted(list(chars))
    chars = ['<pad>', '<unk>'] + chars
    char2idx = {c:i for i,c in enumerate(chars)}
    return char2idx
