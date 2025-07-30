import re
import string

def clean_text(text: str) -> str:
    """
    Clean input text by:
    - Lowercasing
    - Replacing multiple spaces/newlines with a single space
    - Removing unwanted control characters
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Normalize new lines and tabs to space
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Remove non-printable/control characters
    text = ''.join(ch for ch in text if ch in string.printable)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Lowercase all text
    text = text.lower()
    # Strip spaces
    text = text.strip()
    return text


def simple_tokenize(text: str) -> list:
    """
    Basic tokenizer:
    - Splits text on whitespace and punctuations
    - Keeps punctuation as separate tokens (period, comma, question mark, exclamation)
    - Removes any extra spaces
    """
    if not text:
        return []

    # Separate punctuation by spaces
    punctuations = ['.', ',', '!', '?', ';', ':', '(', ')', '"', "'"]
    for p in punctuations:
        text = text.replace(p, f' {p} ')
    # Split by whitespace
    tokens = text.split()
    return tokens


def tokenize_and_clean(text: str) -> list:
    """
    Combines cleaning and tokenizing in one step.
    """
    cleaned = clean_text(text)
    tokens = simple_tokenize(cleaned)
    return tokens


def normalize_text_for_vocab(text: str) -> str:
    """
    Normalize text for vocabulary matching or indexing:
    - Lowercase
    - Strip punctuation and special characters (optional)
    """
    text = text.lower()
    # Remove punctuation but keep spaces
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = text.strip()
    return text


# Optional: Sentence splitter (could be later enhanced with NLP libs)
def split_sentences(text: str) -> list:
    """
    Simple sentence splitter by punctuation.
    """
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]


# Example usage (for testing)
if __name__ == "__main__":
    sample_text = "Hello, world! This is an example: Should tokenize properly.    Newlines,\ntabs,\tand weird   spacing?"
    print("Cleaned text:", clean_text(sample_text))
    print("Tokens:", simple_tokenize(sample_text))
    print("Tokens after clean+tokenize:", tokenize_and_clean(sample_text))
    print("Sentences:", split_sentences(sample_text))
