"""
Model BiLSTM + FastText dan fungsi pelatihannya (dijalankan di GPU Kaggle/Colab).

Arsitektur: Embedding (FastText 300) -> BiLSTM (128 unit) -> Dropout (0,3)
            -> Dense (64, ReLU) -> 6 kelas (softmax dihitung di dalam loss).
Early stopping memakai Macro F1 pada data validasi.
"""
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from src.evaluate import URUTAN_LABEL
from src.features import bangun_kosakata, bangun_matriks_embedding, teks_ke_nomor


def atur_seed(seed):
    """Kunci angka acak agar hasil bisa diulang."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelBiLSTM(nn.Module):
    def __init__(self, matriks_embedding, jumlah_kelas=6, unit=128, dropout=0.3,
                 dense=64, bekukan_embedding=True):
        super().__init__()
        tabel = torch.tensor(matriks_embedding, dtype=torch.float32)
        # bekukan_embedding=True: vektor FastText tidak ikut diubah saat pelatihan
        self.embedding = nn.Embedding.from_pretrained(tabel, freeze=bekukan_embedding, padding_idx=0)
        self.lstm = nn.LSTM(tabel.shape[1], unit, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(unit * 2, dense)
        self.aktivasi = nn.ReLU()
        self.keluaran = nn.Linear(dense, jumlah_kelas)

    def forward(self, nomor, panjang):
        x = self.embedding(nomor)
        # pack_padded_sequence: LSTM tidak membaca bagian pengisi (padding)
        paket = pack_padded_sequence(x, panjang, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(paket)
        gabungan = torch.cat([h[0], h[1]], dim=1)  # akhir bacaan maju + akhir bacaan mundur
        gabungan = self.dropout(gabungan)
        gabungan = self.aktivasi(self.dense(gabungan))
        return self.keluaran(gabungan)  # skor mentah (logit) untuk 6 kelas


def _siapkan_batch(daftar_nomor, indeks, perangkat):
    """Gabungkan beberapa teks menjadi satu batch, diisi nol sampai sama panjang."""
    urutan = [torch.tensor(daftar_nomor[i], dtype=torch.long) for i in indeks]
    panjang = torch.tensor([len(u) for u in urutan])  # tetap di CPU
    nomor = pad_sequence(urutan, batch_first=True, padding_value=0)
    return nomor.to(perangkat), panjang


def _prediksi_indeks(model, daftar_nomor, ukuran_batch, perangkat):
    """Prediksi nomor kelas untuk semua teks."""
    model.eval()
    hasil = []
    with torch.no_grad():
        for mulai in range(0, len(daftar_nomor), ukuran_batch):
            indeks = range(mulai, min(mulai + ukuran_batch, len(daftar_nomor)))
            nomor, panjang = _siapkan_batch(daftar_nomor, indeks, perangkat)
            hasil.extend(model(nomor, panjang).argmax(dim=1).cpu().tolist())
    return hasil


def latih_bilstm_satu_fold(train, val, test, tabel_vektor, kata_ke_baris,
                           kolom_teks="teks_penuh", seed=42, maks_epoch=10,
                           sabar=3, ukuran_batch=32, lr=1e-3, panjang_maks=128):
    """
    Latih BiLSTM untuk satu fold dan satu seed.
    Mengembalikan (prediksi_uji_dalam_teks, info) dengan info berisi epoch terbaik,
    Macro F1 validasi terbaik, jumlah kata berhasil dapat vektor FastText, dan riwayat.
    """
    atur_seed(seed)
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Kosakata dan embedding HANYA dari data latih fold ini (anti data leakage)
    kosakata = bangun_kosakata(train[kolom_teks])
    matriks, ditemukan = bangun_matriks_embedding(kosakata, tabel_vektor, kata_ke_baris, seed=seed)

    # 2) Ubah teks menjadi deretan nomor, label menjadi nomor kelas
    nomor_train = [teks_ke_nomor(t, kosakata, panjang_maks) for t in train[kolom_teks]]
    nomor_val = [teks_ke_nomor(t, kosakata, panjang_maks) for t in val[kolom_teks]]
    nomor_test = [teks_ke_nomor(t, kosakata, panjang_maks) for t in test[kolom_teks]]
    label_ke_id = {label: i for i, label in enumerate(URUTAN_LABEL)}
    y_train = np.array([label_ke_id[l] for l in train["label"]])
    y_val = np.array([label_ke_id[l] for l in val["label"]])

    # 3) Model, optimizer, dan fungsi kesalahan
    model = ModelBiLSTM(matriks).to(perangkat)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    fungsi_loss = nn.CrossEntropyLoss()
    pengacak = np.random.default_rng(seed)

    terbaik = {"skor": -1.0, "epoch": 0, "bobot": None}
    tanpa_kemajuan = 0
    riwayat = []

    for epoch in range(1, maks_epoch + 1):
        model.train()
        urutan_acak = pengacak.permutation(len(nomor_train))
        total_loss = 0.0
        for mulai in range(0, len(urutan_acak), ukuran_batch):
            indeks = urutan_acak[mulai:mulai + ukuran_batch]
            nomor, panjang = _siapkan_batch(nomor_train, indeks, perangkat)
            target = torch.tensor(y_train[indeks], dtype=torch.long, device=perangkat)
            optimizer.zero_grad()
            loss = fungsi_loss(model(nomor, panjang), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indeks)

        # Nilai Macro F1 pada data validasi menentukan model terbaik dan early stopping
        skor_val = f1_score(y_val, _prediksi_indeks(model, nomor_val, ukuran_batch, perangkat),
                            average="macro")
        riwayat.append({"epoch": epoch, "loss_latih": total_loss / len(nomor_train),
                        "macro_f1_val": skor_val})

        if skor_val > terbaik["skor"]:
            terbaik = {"skor": skor_val, "epoch": epoch,
                       "bobot": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            tanpa_kemajuan = 0
        else:
            tanpa_kemajuan += 1
            if tanpa_kemajuan >= sabar:
                break  # early stopping

    # 4) Pakai bobot terbaik untuk memprediksi data uji
    model.load_state_dict(terbaik["bobot"])
    id_prediksi = _prediksi_indeks(model, nomor_test, ukuran_batch, perangkat)
    prediksi = [URUTAN_LABEL[i] for i in id_prediksi]
    info = {"epoch_terbaik": terbaik["epoch"], "macro_f1_val": terbaik["skor"],
            "kata_ditemukan": ditemukan, "ukuran_kosakata": len(kosakata), "riwayat": riwayat}
    return prediksi, info