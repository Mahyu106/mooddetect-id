"""
Penghitung skor untuk semua model (SVM, BiLSTM, IndoBERT).
Metrik utama: Macro F1. Pendamping: accuracy, weighted F1,
serta precision, recall, dan F1 per kelas.
"""
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Urutan kelas dibuat tetap agar semua tabel dan grafik konsisten
URUTAN_LABEL = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def hitung_metrik(y_benar, y_prediksi, labels=URUTAN_LABEL):
    """Hitung semua metrik untuk satu fold; kembalikan dalam bentuk dictionary."""
    presisi, recall, f1_kelas, _ = precision_recall_fscore_support(
        y_benar, y_prediksi, labels=labels, zero_division=0
    )
    hasil = {
        "accuracy": accuracy_score(y_benar, y_prediksi),
        "macro_f1": f1_score(y_benar, y_prediksi, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_benar, y_prediksi, labels=labels, average="weighted", zero_division=0),
    }
    for i, nama in enumerate(labels):
        hasil[f"precision_{nama}"] = presisi[i]
        hasil[f"recall_{nama}"] = recall[i]
        hasil[f"f1_{nama}"] = f1_kelas[i]
    return hasil


def hitung_confusion(y_benar, y_prediksi, labels=URUTAN_LABEL):
    """Matriks kebingungan: baris = emosi asli, kolom = emosi prediksi."""
    return confusion_matrix(y_benar, y_prediksi, labels=labels)