"""
Pembuatan 5 fold stratified yang sama untuk semua model.

Di tiap fold:
- test  : 1/5 data (data uji)
- train : sisanya, setelah 10% diambil untuk validation
- val   : 10% dari data latih (untuk early stopping dan menentukan ambang)
Semua pembagian stratified (proporsi tiap emosi dijaga).
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def buat_fold(df, kolom_label="label", n_fold=5, porsi_val=0.10, seed=42):
    """
    Kembalikan tabel pembagian dengan kolom:
    fold (0-4), indeks (nomor baris di df), peran (train / val / test).
    """
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    potongan = []
    for nomor_fold, (idx_latih, idx_uji) in enumerate(skf.split(df, df[kolom_label])):
        # Ambil 10% dari data latih sebagai validation, tetap stratified
        idx_train, idx_val = train_test_split(
            idx_latih,
            test_size=porsi_val,
            stratify=df[kolom_label].iloc[idx_latih],
            random_state=seed,
        )
        for peran, idx in (("train", idx_train), ("val", idx_val), ("test", idx_uji)):
            potongan.append(pd.DataFrame({"fold": nomor_fold, "indeks": idx, "peran": peran}))
    return pd.concat(potongan, ignore_index=True)


def ambil_fold(df, tabel_fold, nomor_fold):
    """Ambil tiga potongan data (train, val, test) untuk satu fold."""
    bagian = tabel_fold[tabel_fold["fold"] == nomor_fold]
    hasil = {}
    for peran in ("train", "val", "test"):
        idx = bagian.loc[bagian["peran"] == peran, "indeks"].to_numpy()
        hasil[peran] = df.iloc[idx].reset_index(drop=True)
    return hasil["train"], hasil["val"], hasil["test"]