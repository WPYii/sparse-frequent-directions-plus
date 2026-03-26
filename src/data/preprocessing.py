from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp

class Preprocessor:
    def __init__(self, lowercase, stop_words, max_features):
        self.vectorizer = CountVectorizer(
            binary=True,
            lowercase=lowercase,
            stop_words=stop_words,
            max_features=max_features
        )

    def build_sparse_binary_matrix(self, documents):
        X = self.vectorizer.fit_transform(documents).tocsr()
        return X

    def transform(self, X, d):
        A = X.transpose().tocsr()
        return A[:, :d]
    
    def get_vocab_size(self):
        return len(self.vectorizer.vocabulary_)