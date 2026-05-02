# utility functions - can not execute
import math
import re
from collections import Counter

import nltk
import numpy as np
from gensim.models import Phrases
from gensim.models.phrases import Phraser

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

FIELDS = ["name", "tags", "description", "ingredients"] # "steps" excluded via experiment - check 'field_experiment.py' file for details
STOPWORDS = set(stopwords.words('english'))
DIGITS = re.compile(r'\d')
lemmatizer = WordNetLemmatizer()

def preprocess(doc):
    words = re.findall(r'\b[a-z]+\b', doc.lower())  # tokenize + lowercase + punctuation in one pass
    words = [w for w in words if w not in STOPWORDS]
    words = [w for w in words if not DIGITS.search(w)]
    words = [lemmatizer.lemmatize(w) for w in words]  # noun default, fast
    return words

def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)

def doc_to_text(doc):
    parts = []
    for field in FIELDS:
        if field in doc:
            parts.append(normalize_text(doc[field]))
    return " ".join(parts)

def build_vocab(tokenized_docs, min_df=6, max_df_ratio=0.5, max_vocab=50000):
    df = Counter()
    for tokens in tokenized_docs:
        for term in set(tokens):
            df[term] += 1

    doc_count = len(tokenized_docs)
    max_df = int(max_df_ratio * doc_count)
    terms = [t for t, d in df.items() if d >= min_df and d <= max_df]
    terms.sort(key=lambda t: (-df[t], t))
    if len(terms) > max_vocab:
        terms = terms[:max_vocab]

    vocab = {term: idx for idx, term in enumerate(terms)}
    idf = np.zeros(len(vocab), dtype=np.float32)
    for term, idx in vocab.items():
        idf[idx] = math.log((doc_count + 1) / (df[term] + 1)) + 1.0
    return vocab, idf, df

def vectorize(tokens, vocab, idf):
    counts = Counter(tokens)
    vec = {}
    for term, count in counts.items():
        idx = vocab.get(term)
        if idx is None:
            continue
        tf = 1.0 + math.log(count)
        vec[idx] = tf * float(idf[idx])
    norm = math.sqrt(sum(value * value for value in vec.values()))
    if norm > 0:
        for key in list(vec.keys()):
            vec[key] /= norm
    return vec

def dot_sparse(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(idx, 0.0) for idx, value in a.items())

def vectorize_query(query, bigram, vocab, idf):
    tokens = preprocess(query)
    tokens = bigram[tokens]
    return vectorize(tokens, vocab, idf)

def tokenize_docs(doc_texts):
    tokenized_docs = []
    for text in doc_texts:
        tokenized_docs.append(preprocess(text))
    print("Tokenized docs.")
    return tokenized_docs

def train_bigram_model(tokenized_docs):
    # after preprocessing all docs into token lists
    phrases = Phrases(tokenized_docs, min_count=5, threshold=3.0)
    bigram = Phraser(phrases)
    print("Trained bigram model.")
    return bigram
