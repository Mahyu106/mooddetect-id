"""
Pelatihan IndoBERT (indobenchmark/indobert-base-p1) untuk klasifikasi 6 emosi.
Dijalankan di GPU Kaggle/Colab. Memakai loop PyTorch manual.

Hyperparameter standar: lr 2e-5, hingga 5 epoch, early stopping patience 2
(Macro F1 validasi), weight decay 0,01, warmup 0,1, fp16, max_length 128,
batch fisik 8 x akumulasi gradien 2 = batch efektif 16.
"""
import gc
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluate import URUTAN_LABEL

NAMA_MODEL = "indobenchmark/indobert-base-p1"


def atur_seed(seed):
    """Kunci angka acak agar hasil bisa diulang."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _buat_batch(daftar_id, indeks, pad_id, perangkat):
    """Gabungkan beberapa teks menjadi satu batch (diisi sampai sama panjang)."""
    panjang = max(len(daftar_id[i]) for i in indeks)
    nomor = torch.full((len(indeks), panjang), pad_id, dtype=torch.long)
    mask = torch.zeros((len(indeks), panjang), dtype=torch.long)
    for baris, i in enumerate(indeks):
        n = len(daftar_id[i])
        nomor[baris, :n] = torch.tensor(daftar_id[i])
        mask[baris, :n] = 1  # 1 = token asli, 0 = pengisi
    return nomor.to(perangkat), mask.to(perangkat)


def _prediksi_indeks(model, daftar_id, pad_id, perangkat, pakai_fp16, ukuran_batch=32):
    """Prediksi nomor kelas untuk semua teks."""
    model.eval()
    hasil = []
    pakai_amp = pakai_fp16 and perangkat.type == "cuda"
    with torch.no_grad():
        for mulai in range(0, len(daftar_id), ukuran_batch):
            indeks = list(range(mulai, min(mulai + ukuran_batch, len(daftar_id))))
            nomor, mask = _buat_batch(daftar_id, indeks, pad_id, perangkat)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=pakai_amp):
                logit = model(input_ids=nomor, attention_mask=mask).logits
            hasil.extend(logit.argmax(dim=1).cpu().tolist())
    return hasil


def latih_indobert_satu_fold(train, val, test, kolom_teks="teks_minimal", seed=42,
                             lr=2e-5, maks_epoch=5, sabar=2, panjang_maks=128,
                             batch_fisik=8, akumulasi=2, weight_decay=0.01,
                             warmup=0.1, pakai_fp16=True, pakai_class_weight=False,
                             nama_model=NAMA_MODEL):
    """
    Latih IndoBERT untuk satu fold dan satu seed.
    Mengembalikan (prediksi_uji_dalam_teks, info).
    """
    atur_seed(seed)
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pakai_amp = pakai_fp16 and perangkat.type == "cuda"

    # 1) Tokenisasi semua teks (dipotong sampai panjang_maks token)
    tokenizer = AutoTokenizer.from_pretrained(nama_model)

    def tokenisasi(teks):
        return tokenizer(list(teks), truncation=True, max_length=panjang_maks)["input_ids"]

    id_train, id_val, id_test = (tokenisasi(d[kolom_teks]) for d in (train, val, test))
    pad_id = tokenizer.pad_token_id

    label_ke_id = {label: i for i, label in enumerate(URUTAN_LABEL)}
    y_train = np.array([label_ke_id[l] for l in train["label"]])
    y_val = np.array([label_ke_id[l] for l in val["label"]])

    # 2) Model IndoBERT + lapisan klasifikasi 6 kelas (peringatan 'newly initialized' itu normal)
    model = AutoModelForSequenceClassification.from_pretrained(
        nama_model, num_labels=len(URUTAN_LABEL)).to(perangkat)

    # 3) Optimizer AdamW; weight decay tidak dikenakan pada bias dan LayerNorm
    tanpa_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    grup = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in tanpa_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in tanpa_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grup, lr=lr)

    # 4) Jadwal learning rate: naik linear selama warmup, lalu turun linear ke nol
    batch_efektif = batch_fisik * akumulasi
    langkah_per_epoch = -(-len(id_train) // batch_efektif)  # pembulatan ke atas
    total_langkah = maks_epoch * langkah_per_epoch
    langkah_warmup = int(warmup * total_langkah)

    def faktor_lr(langkah):
        if langkah < langkah_warmup:
            return langkah / max(1, langkah_warmup)
        return max(0.0, (total_langkah - langkah) / max(1, total_langkah - langkah_warmup))

    penjadwal = torch.optim.lr_scheduler.LambdaLR(optimizer, faktor_lr)
    penskala = torch.amp.GradScaler("cuda", enabled=pakai_amp)  # untuk fp16

    # 5) Fungsi kesalahan; bobot kelas opsional (untuk variasi eksperimen)
    if pakai_class_weight:
        jumlah = np.bincount(y_train, minlength=len(URUTAN_LABEL))
        bobot = torch.tensor(len(y_train) / (len(URUTAN_LABEL) * jumlah),
                             dtype=torch.float32, device=perangkat)
        fungsi_loss = nn.CrossEntropyLoss(weight=bobot)
    else:
        fungsi_loss = nn.CrossEntropyLoss()

    pengacak = np.random.default_rng(seed)
    terbaik = {"skor": -1.0, "epoch": 0, "bobot": None}
    tanpa_kemajuan = 0
    riwayat = []

    for epoch in range(1, maks_epoch + 1):
        mulai_epoch = time.time()
        model.train()
        urutan_acak = pengacak.permutation(len(id_train))
        kelompok = [urutan_acak[i:i + batch_fisik] for i in range(0, len(urutan_acak), batch_fisik)]
        total_loss = 0.0
        optimizer.zero_grad()
        for k, indeks in enumerate(kelompok, start=1):
            nomor, mask = _buat_batch(id_train, indeks, pad_id, perangkat)
            target = torch.tensor(y_train[indeks], dtype=torch.long, device=perangkat)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=pakai_amp):
                logit = model(input_ids=nomor, attention_mask=mask).logits
            loss = fungsi_loss(logit.float(), target)
            total_loss += loss.item() * len(indeks)
            penskala.scale(loss / akumulasi).backward()
            # Perbarui bobot tiap 'akumulasi' kelompok kecil (atau di kelompok terakhir)
            if k % akumulasi == 0 or k == len(kelompok):
                penskala.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # batasi lonjakan gradien
                penskala.step(optimizer)
                penskala.update()
                penjadwal.step()
                optimizer.zero_grad()

        # Macro F1 pada data validasi menentukan model terbaik dan early stopping
        skor_val = f1_score(y_val, _prediksi_indeks(model, id_val, pad_id, perangkat, pakai_fp16),
                            average="macro")
        riwayat.append({"epoch": epoch, "loss_latih": total_loss / len(id_train),
                        "macro_f1_val": skor_val, "detik": time.time() - mulai_epoch})

        if skor_val > terbaik["skor"]:
            terbaik = {"skor": skor_val, "epoch": epoch,
                       "bobot": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            tanpa_kemajuan = 0
        else:
            tanpa_kemajuan += 1
            if tanpa_kemajuan >= sabar:
                break  # early stopping

    # 6) Pakai bobot terbaik untuk memprediksi data uji
    model.load_state_dict(terbaik["bobot"])
    id_prediksi = _prediksi_indeks(model, id_test, pad_id, perangkat, pakai_fp16)
    prediksi = [URUTAN_LABEL[i] for i in id_prediksi]
    info = {"epoch_terbaik": terbaik["epoch"], "macro_f1_val": terbaik["skor"], "riwayat": riwayat}

    # 7) Bebaskan memori GPU agar putaran berikutnya tidak kehabisan memori
    del model, optimizer, terbaik
    gc.collect()
    torch.cuda.empty_cache()
    return prediksi, info