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

# ======================================================================
# Bagian untuk BiLSTM + FastText
# ======================================================================
from collections import Counter

import numpy as np

PAD, UNK = "<pad>", "<unk>"  # nomor 0 = padding (pengisi), nomor 1 = kata tak dikenal


def bangun_kosakata(daftar_teks):
    """Buat kosakata (kata -> nomor). PENTING: hanya dari data latih satu fold."""
    hitung = Counter(kata for teks in daftar_teks for kata in str(teks).split())
    kata_ke_nomor = {PAD: 0, UNK: 1}
    for kata in hitung:
        if kata not in kata_ke_nomor:
            kata_ke_nomor[kata] = len(kata_ke_nomor)
    return kata_ke_nomor


def teks_ke_nomor(teks, kata_ke_nomor, panjang_maks=128):
    """Ubah satu teks menjadi deretan nomor; kata asing menjadi UNK, dipotong sampai panjang_maks."""
    nomor = [kata_ke_nomor.get(kata, 1) for kata in str(teks).split()][:panjang_maks]
    return nomor if nomor else [1]  # teks kosong diberi satu UNK


def muat_fasttext(jalur):
    """
    Baca vektor FastText format gensim lama secara manual (tanpa KeyedVectors.load).
    Mengembalikan (daftar_kata, tabel_vektor, kata_ke_baris).
    """
    from gensim.utils import unpickle  # diimpor di sini agar modul tetap ringan di laptop

    obj = unpickle(jalur)
    daftar_kata = obj.__dict__["index2word"]
    tabel_vektor = np.load(f"{jalur}.vectors.npy")
    kata_ke_baris = {kata: i for i, kata in enumerate(daftar_kata)}
    return daftar_kata, tabel_vektor, kata_ke_baris


def bangun_matriks_embedding(kata_ke_nomor, tabel_vektor, kata_ke_baris, seed=42):
    """
    Buat tabel embedding untuk kosakata satu fold.
    Kata yang punya vektor FastText memakai vektor itu; sisanya vektor acak kecil.
    Baris 0 (PAD) berisi nol. Mengembalikan (matriks, jumlah_kata_ditemukan).
    """
    rng = np.random.default_rng(seed)
    dim = tabel_vektor.shape[1]
    matriks = rng.normal(0, 0.1, (len(kata_ke_nomor), dim)).astype("float32")
    matriks[0] = 0.0  # PAD = nol
    ditemukan = 0
    for kata, nomor in kata_ke_nomor.items():
        baris = kata_ke_baris.get(kata)
        if baris is not None:
            matriks[nomor] = tabel_vektor[baris]
            ditemukan += 1
    return matriks, ditemukan