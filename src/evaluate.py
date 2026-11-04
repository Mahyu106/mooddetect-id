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

# ======================================================================
# Penggabungan dan ringkasan hasil semua model (dijalankan di laptop)
# ======================================================================
import glob
import os

import pandas as pd

KOLOM_METRIK = ["macro_f1", "accuracy", "weighted_f1"]


def muat_semua_hasil(folder="../reports/tables"):
    """
    Gabungkan semua tabel hasil per fold (SVM, BiLSTM, dan semua eksperimen IndoBERT)
    menjadi satu tabel. Nama model diambil dari kolom 'model' di tiap file.
    """
    berkas = sorted(glob.glob(os.path.join(folder, "*per_fold*.csv")))
    tabel = pd.concat([pd.read_csv(b) for b in berkas], ignore_index=True)
    if "seed" not in tabel.columns:
        tabel["seed"] = float("nan")  # SVM bersifat deterministik, tidak punya seed
    return tabel


def matriks_per_fold(tabel, metrik="macro_f1"):
    """Tabel baris = fold, kolom = model; nilai = metrik dirata-rata antar seed."""
    return tabel.groupby(["fold", "model"])[metrik].mean().unstack("model")


def ringkas_per_model(tabel):
    """
    Ringkasan tiap model: rata-rata dan simpangan baku antar fold untuk Macro F1,
    accuracy, dan weighted F1. Seed dirata-rata dulu di dalam tiap fold.
    """
    per_fold = tabel.groupby(["model", "fold"])[KOLOM_METRIK].mean().reset_index()
    ringkas = per_fold.groupby("model")[KOLOM_METRIK].agg(["mean", "std"])
    ringkas.columns = [f"{kolom}_{statistik}" for kolom, statistik in ringkas.columns]
    return ringkas.sort_values("macro_f1_mean", ascending=False).reset_index()