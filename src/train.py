"""
Fungsi pelatihan model.
Saat ini berisi SVM + TF-IDF; BiLSTM dan IndoBERT akan ditambahkan nanti.
"""
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from src.features import tfidf_untuk_fold

# Nilai C yang dicoba (dipilih memakai data validasi)
NILAI_C = [0.1, 1, 10]


def latih_svm_satu_fold(train, val, test, kolom_teks="teks_penuh",
                        nilai_c=NILAI_C, seed=42):
    """
    Latih SVM untuk satu fold.
    1) TF-IDF dipelajari hanya dari data latih.
    2) Tiap nilai C dilatih di data latih, lalu dinilai di data validasi.
    3) C terbaik (Macro F1 validasi tertinggi) dipakai memprediksi data uji.
    Mengembalikan (prediksi_uji, c_terbaik, macro_f1_validasi).
    """
    X_train, X_val, X_test, _ = tfidf_untuk_fold(
        train[kolom_teks], val[kolom_teks], test[kolom_teks]
    )

    terbaik = None
    for c in nilai_c:
        model = LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=10000)
        model.fit(X_train, train["label"])
        skor_val = f1_score(val["label"], model.predict(X_val), average="macro")
        if terbaik is None or skor_val > terbaik["skor_val"]:
            terbaik = {"c": c, "skor_val": skor_val, "model": model}

    prediksi_uji = terbaik["model"].predict(X_test)
    return prediksi_uji, terbaik["c"], terbaik["skor_val"]