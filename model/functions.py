import nltk
import string
import re

from collections import Counter

nltk.download("stopwords")
from nltk.corpus import stopwords

stop = set(stopwords.words("english"))


def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


def counter_word(text_array):
    count = Counter()
    for text in text_array:
        for word in text.split():
            count[word] += 1
    return count


def decode(sequence, word_index):
    return " ".join([word_index.get(idx, "?") for idx in sequence])
