# SkipGram News Experiment

##  Deskripsi Proyek
Proyek ini mengimplementasikan model **SkipGram** untuk pembelajaran representasi kata (*word embeddings*) menggunakan dataset teks yang diambil dari **NewsAPI**. Tujuan utama eksperimen ini adalah untuk menguji pengaruh variasi **window size** dan **dimensi embedding** terhadap kualitas representasi kata yang dihasilkan.

Hasil eksperimen dievaluasi dengan mengukur kedekatan kata-kata berdasarkan cosine similarity untuk melihat hubungan semantik yang dihasilkan oleh model.

---

## Cara Menggunakan

### Instalasi

Clone repository:
```bash
git clone https://github.com/rismawidiyanti/skipgram-news-experiment.git
cd skipgram-news-experiment
```

Install dependensi:
```bash
pip install -r requirements.txt
```

### Menjalankan Kode

1. Tambahkan API Key pada file `skipgram_news.py`:
   ```python
   api_key = "   "
   ```

2. Jalankan program:
```bash
python skipgram_news.py
```

### Dependensi
- `numpy`
- `requests`

---

##  Penjelasan Parameter Eksperimen

### Variasi Window Size
- **1** → Mengambil 1 kata di sekitar kata target.
- **2** → Mengambil 2 kata di sekitar kata target.
- **3** → Mengambil 3 kata di sekitar kata target.

### Dimensi Embedding
- **10 dimensi** → Representasi kata dalam ruang berdimensi 10.
- **20** → Representasi kata dalam 20 dimensi.
- **50** → Representasi kata dalam 50 dimensi.

### Hyperparameter Lainnya
- **Learning rate**: 0.01
- **Jumlah Epoch**: 5

---

## Hasil Eksperimen dan Analisis

### Observasi Awal
- **Loss** selama pelatihan relatif stagnan di nilai tinggi (~20), menunjukkan model belum mencapai konvergensi optimal.
- **Cosine similarity** antara kata sering bernilai 1, menandakan embedding masih kurang membedakan hubungan semantik antar kata secara efektif.
- Kata tertentu tidak ditemukan dalam vocabulary karena keterbatasan dataset.

### Analisis Window Size dan Embedding Dimension
- Window size kecil (1) menghasilkan konteks terbatas, embedding cenderung kurang kaya secara semantik.
- Window size lebih besar (2 dan 3) mulai memperlihatkan sedikit perbedaan representasi kata, tetapi masih belum optimal karena dataset terbatas.
- Dimensi embedding yang lebih besar tidak secara signifikan meningkatkan kualitas embedding jika data tidak cukup atau pelatihan terlalu singkat.

---

## Kesimpulan
Eksperimen menunjukkan bahwa variasi window size dan dimensi embedding sangat berpengaruh terhadap representasi kata yang dihasilkan. Namun, hasil awal menunjukkan perlunya peningkatan dalam preprocessing data dan tuning hyperparameter untuk mencapai embedding yang optimal dan semantik lebih jelas. Langkah perbaikan di atas dapat meningkatkan performa dan menghasilkan embedding yang lebih representatif secara semantik.

---

## Dependensi
- numpy
- requests

---

Terima kasih! 
```

