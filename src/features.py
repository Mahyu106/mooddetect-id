"""
Pembuatan fitur TF-IDF untuk SVM.

PENTING (anti data leakage): TF-IDF hanya boleh di-fit pada data latih
di tiap fold. Data validasi dan data uji hanya di-transform.
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def buat_tfidf():
    """TF-IDF sesuai rancangan: unigram + bigram, maksimal 10.000 fitur."""
    return TfidfVectorizer(ngram_range=(1, 2), max_features=10000)


def tfidf_untuk_fold(teks_train, teks_val, teks_test):
    """
    Pelajari kosakata hanya dari teks_train, lalu ubah ketiga bagian menjadi
    matriks angka. Mengembalikan (X_train, X_val, X_test, vectorizer).
    """
    vektorizer = buat_tfidf()
    X_train = vektorizer.fit_transform(teks_train)  # fit HANYA di data latih
    X_val = vektorizer.transform(teks_val)          # val: hanya transform
    X_test = vektorizer.transform(teks_test)        # test: hanya transform
    return X_train, X_val, X_test, vektorizer