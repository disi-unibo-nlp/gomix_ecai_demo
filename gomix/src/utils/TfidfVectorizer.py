from gensim import corpora
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
from typing import Callable, Iterable, List


class TfidfVectorizer:
    def __init__(self, find_training_docs: Callable[[], Iterable[str]]):
        """
        Train TF-IDF vectorizer on given documents.
        """
        self._dictionary = corpora.Dictionary(simple_preprocess(doc) for doc in find_training_docs())
        self._dictionary.filter_extremes(no_below=50, no_above=0.8)
        self._tfidf = TfidfModel(self._dictionary.doc2bow(simple_preprocess(doc)) for doc in find_training_docs())

    def vectorize(self, doc: str) -> List[float]:
        """
        Use trained TF-IDF vectorizer to vectorize given document.
        """
        vec_bow = self._dictionary.doc2bow(simple_preprocess(doc))
        vec_tfidf_sparse = self._tfidf[vec_bow]
        vec_tfidf_dense = [0.] * len(self._dictionary)
        for idx, value in vec_tfidf_sparse:
            vec_tfidf_dense[idx] = value
        return vec_tfidf_dense
