from model.functions import remove_URL, remove_punct, remove_stopwords, counter_word, decode
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.sequence import pad_sequences


class RemoveURL(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):

        X_processed = [remove_URL(text) for text in X]
        return X_processed


class RemovePunctuations(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):

        X_processed = [remove_punct(text) for text in X]
        return X_processed


class RemoveStopWords(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):

        X_processed = [remove_stopwords(text) for text in X]

        return X_processed


class TextToSequence(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer) -> None:
        super(TextToSequence, self).__init__()
        self.tokenizer = tokenizer

    def fit(self, X, y):
        return self

    def transform(self, X):
        text_sequences = self.tokenizer.texts_to_sequences(X)

        return text_sequences


class PadSequences(BaseEstimator, TransformerMixin):
    def __init__(self, max_length) -> None:
        super(PadSequences, self).__init__()
        self.max_length = max_length

    def fit(self, X, y):
        return self

    def transform(self, X):
        X_padded = pad_sequences(X, maxlen=self.max_length, padding="post", truncating="post")

        return X_padded
