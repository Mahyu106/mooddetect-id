"""
Pelatihan model final IndoBERT (konfigurasi E3) untuk deployment.
Dilatih di data latih final (90%), epoch terbaik dipilih dari Macro F1 pada hold-out.
Menyimpan model + tokenizer (format Hugging Face) dan probabilitas hold-out.
Dijalankan di GPU Kaggle.
"""
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluate import URUTAN_LABEL
from src.indobert import NAMA_MODEL, _buat_batch, atur_seed


def _probabilitas(model, daftar_id, pad_id, perangkat, pakai_fp16, ukuran_batch=32):
    """Softmax tiap emosi untuk semua teks (kolom mengikuti URUTAN_LABEL)."""
    model.eval()
    hasil = []
    pakai_amp = pakai_fp16 and perangkat.type == "cuda"
    with torch.no_grad():
        for mulai in range(0, len(daftar_id), ukuran_batch):
            indeks = list(range(mulai, min(mulai + ukuran_batch, len(daftar_id))))
            nomor, mask = _buat_batch(daftar_id, indeks, pad_id, perangkat)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=pakai_amp):
                logit = model(input_ids=nomor, attention_mask=mask).logits
            hasil.append(torch.softmax(logit.float(), dim=1).cpu().numpy())
    return np.vstack(hasil)


def latih_model_final(train, holdout, folder_keluar, kolom_teks="teks_minimal", seed=42,
                      lr=2e-5, maks_epoch=5, sabar=2, panjang_maks=128,
                      batch_fisik=8, akumulasi=2, weight_decay=0.01,
                      warmup=0.1, pakai_fp16=True, nama_model=NAMA_MODEL):
    """Latih model final, simpan ke folder_keluar, kembalikan (tabel_probabilitas_holdout, info)."""
    atur_seed(seed)
    folder_keluar = Path(folder_keluar)
    folder_keluar.mkdir(parents=True, exist_ok=True)
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pakai_amp = pakai_fp16 and perangkat.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(nama_model)

    def tokenisasi(teks):
        return tokenizer(list(teks), truncation=True, max_length=panjang_maks)["input_ids"]

    id_train, id_hold = tokenisasi(train[kolom_teks]), tokenisasi(holdout[kolom_teks])
    pad_id = tokenizer.pad_token_id

    label_ke_id = {label: i for i, label in enumerate(URUTAN_LABEL)}
    y_train = np.array([label_ke_id[l] for l in train["label"]])
    y_hold = np.array([label_ke_id[l] for l in holdout["label"]])

    # Simpan pemetaan label di config supaya predict.py tidak perlu menebak urutan
    model = AutoModelForSequenceClassification.from_pretrained(
        nama_model, num_labels=len(URUTAN_LABEL),
        id2label={i: l for i, l in enumerate(URUTAN_LABEL)},
        label2id=label_ke_id).to(perangkat)

    tanpa_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    grup = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in tanpa_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in tanpa_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grup, lr=lr)

    batch_efektif = batch_fisik * akumulasi
    langkah_per_epoch = -(-len(id_train) // batch_efektif)
    total_langkah = maks_epoch * langkah_per_epoch
    langkah_warmup = int(warmup * total_langkah)

    def faktor_lr(langkah):
        if langkah < langkah_warmup:
            return langkah / max(1, langkah_warmup)
        return max(0.0, (total_langkah - langkah) / max(1, total_langkah - langkah_warmup))

    penjadwal = torch.optim.lr_scheduler.LambdaLR(optimizer, faktor_lr)
    penskala = torch.amp.GradScaler("cuda", enabled=pakai_amp)
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
            if k % akumulasi == 0 or k == len(kelompok):
                penskala.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                penskala.step(optimizer)
                penskala.update()
                penjadwal.step()
                optimizer.zero_grad()

        prob = _probabilitas(model, id_hold, pad_id, perangkat, pakai_fp16)
        skor = f1_score(y_hold, prob.argmax(axis=1), average="macro")
        riwayat.append({"epoch": epoch, "loss_latih": total_loss / len(id_train),
                        "macro_f1_holdout": skor, "detik": time.time() - mulai_epoch})
        print(f"Epoch {epoch}: loss {total_loss / len(id_train):.4f} | Macro F1 holdout {skor:.4f}")

        if skor > terbaik["skor"]:
            terbaik = {"skor": skor, "epoch": epoch,
                       "bobot": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            tanpa_kemajuan = 0
        else:
            tanpa_kemajuan += 1
            if tanpa_kemajuan >= sabar:
                break

    # Pakai bobot terbaik, hitung probabilitas hold-out, lalu simpan model
    model.load_state_dict(terbaik["bobot"])
    prob_final = _probabilitas(model, id_hold, pad_id, perangkat, pakai_fp16)
    model.save_pretrained(folder_keluar)
    tokenizer.save_pretrained(folder_keluar)

    tabel = pd.DataFrame(prob_final, columns=[f"prob_{l}" for l in URUTAN_LABEL])
    tabel.insert(0, "label_asli", holdout["label"].values)
    tabel.insert(1, "indeks", holdout["indeks"].values if "indeks" in holdout.columns else range(len(holdout)))
    info = {"epoch_terbaik": terbaik["epoch"], "macro_f1_holdout": terbaik["skor"], "riwayat": riwayat}

    del model, optimizer, terbaik
    gc.collect()
    torch.cuda.empty_cache()
    return tabel, info