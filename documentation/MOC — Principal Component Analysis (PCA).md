#  MOC — Principal Component Analysis (PCA)

> **Tujuan:** Memahami teori dan implementasi PCA untuk reduksi dimensi dan visualisasi data berdimensi tinggi.
> 
> **Status:** 🟡 In Progress **Terakhir diupdate:** {27/05/2026} **Tag:** #moc #ml #unsupervised #pca #dimensionality-reduction

---

##  Navigasi Cepat

- [[# Fondasi Matematika]]
- [[#📐 Teori Inti PCA]]
- [[#⚙️ Algoritma & Langkah-langkah]]
- [[#📊 Evaluasi & Interpretasi]]
- [[#🛠️ Implementasi]]
- [[#📈 Visualisasi]]
- [[#🔁 PCA vs Metode Lain]]
- [[#📚 Referensi & Sumber]]

---

##  Fondasi Matematika

> Sebelum memahami PCA, kuasai fondasi ini terlebih dahulu.

- [[Vektor dan Ruang Vektor]] — basis dari representasi data multidimensi
- [[Matriks dan Operasi Matriks]] — perkalian, transpos, invers
- [[Eigenvalue dan Eigenvector]] ⭐ — konsep paling krusial dalam PCA
- [[Dekomposisi Matriks (SVD)]] — Singular Value Decomposition, alternatif path ke PCA
- [[Covariance dan Correlation Matrix]] ⭐ — mengukur hubungan antar fitur
- [[Distribusi Data Multivariat]] — memahami data berdimensi tinggi secara statistik
- [[Proyeksi Linear]] — intuisi geometri dari PCA

---

## 📐 Teori Inti PCA

> Inti dari apa yang dilakukan PCA dan mengapa ia bekerja.

- [[Apa itu Dimensionality Reduction]] ⭐ — masalah yang dipecahkan PCA
- [[Apa itu PCA]] ⭐ — definisi, tujuan, dan intuisi
- [[Principal Components adalah Apa]] — memahami "komponen utama"
- [[Variance dan Arah Maksimum Variance]] — PCA mencari arah dengan variance terbesar
- [[Asumsi-asumsi PCA]] — linearitas, skala, distribusi normal
- [[PCA sebagai Rotasi Koordinat]] — intuisi geometri yang kuat
- [[Curse of Dimensionality]] — mengapa dimensi tinggi itu bermasalah

---

## ⚙️ Algoritma & Langkah-langkah

> Cara PCA bekerja step by step.

- [[Langkah 1 — Standarisasi Data (StandardScaler)]] ⭐
- [[Langkah 2 — Hitung Covariance Matrix]]
- [[Langkah 3 — Hitung Eigenvalue dan Eigenvector]]
- [[Langkah 4 — Urutkan Eigenvector (Principal Components)]]
- [[Langkah 5 — Proyeksikan Data ke Principal Components]]
- [[Hubungan PCA dengan SVD]] — dua cara berbeda, hasil yang sama
- [[PCA Manual vs PCA dengan Library]] — perhitungan dari scratch vs scikit-learn

---

## 📊 Evaluasi & Interpretasi

> Bagaimana tahu apakah PCA kita sudah baik?

- [[Explained Variance Ratio]] ⭐ — seberapa banyak informasi yang dipertahankan
- [[Cumulative Explained Variance]] — berapa PC yang cukup?
- [[Scree Plot]] ⭐ — visualisasi untuk memilih jumlah komponen
- [[Kaiser Criterion]] — aturan eigenvalue > 1
- [[Elbow Method pada PCA]] — kapan kurva mulai mendatar
- [[Loadings dan Factor Scores]] — apa kontribusi tiap fitur ke PC
- [[Interpretasi Principal Component]] — apa "makna" dari tiap PC

---

## 🛠️ Implementasi

> Catatan teknis dan kodingan.

- [[Setup Environment untuk PCA]] — library yang dibutuhkan
- [[PCA dengan Scikit-learn]] ⭐ — `sklearn.decomposition.PCA`
- [[Preprocessing Sebelum PCA]] ⭐ — StandardScaler wajib dipakai
- [[Menentukan Jumlah Komponen n_components]]
- [[PCA Invers Transform — Rekonstruksi Data]]
- [[PCA untuk Data Gambar (MNIST)]] — contoh high-dimensional case
- [[Tips dan Gotchas PCA]] — kesalahan umum yang harus dihindari

---

## 📈 Visualisasi

> Cara menyajikan hasil PCA agar mudah dipahami.

- [[Visualisasi 2D PCA Plot]] ⭐ — scatter plot PC1 vs PC2
- [[Visualisasi 3D PCA Plot]] — dengan Plotly
- [[Scree Plot]] ⭐ — (lihat juga di Evaluasi)
- [[Biplot — Loading Plot + Score Plot]] — melihat kontribusi fitur
- [[Heatmap Covariance Matrix]] — sebelum dan sesudah PCA
- [[PCA Graph View di berbagai Dataset]] — Iris, MNIST, Breast Cancer

---

## 🔁 PCA vs Metode Lain

> Konteks: PCA bukan satu-satunya metode reduksi dimensi.

- [[PCA vs t-SNE]] — linear vs non-linear, interpretasi vs visualisasi
- [[PCA vs UMAP]] — perbandingan modern
- [[PCA vs LDA (Linear Discriminant Analysis)]] — unsupervised vs supervised
- [[PCA vs Autoencoder]] — klasik vs deep learning
- [[Kapan Pakai PCA, Kapan Tidak]]

---

## 📚 Referensi & Sumber

### Paper

- [[Paper — Shlens 2014 — A Tutorial on PCA]] — https://arxiv.org/abs/1404.1100
- [[Paper — Jolliffe & Cadima 2016]] — https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202

### Artikel & Tutorial

- [[Artikel — Scikit-learn PCA Docs]] — https://scikit-learn.org/stable/modules/decomposition.html#pca
- [[Artikel — Towards Data Science PCA]] — https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c

### Video

- [[Video — StatQuest PCA Part 1]] — https://www.youtube.com/watch?v=FgakZw6K1QQ
- [[Video — StatQuest PCA Part 2]] — https://www.youtube.com/watch?v=oRvgq966sz0
- [[Video — 3Blue1Brown Essence of Linear Algebra]] — https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

### Dataset

- [[Dataset — Iris (sklearn)]]
- [[Dataset — Breast Cancer Wisconsin (sklearn)]]
- [[Dataset — MNIST Digits (sklearn)]]
- [[Dataset — Wine Quality (UCI)]]

---

## ✅ Progress Tracker

```
FONDASI MATEMATIKA
[ ] Eigenvalue dan Eigenvector
[ ] Covariance Matrix
[ ] Proyeksi Linear

TEORI INTI
[ ] Apa itu PCA
[ ] Variance dan arah maksimum
[ ] Asumsi-asumsi PCA

ALGORITMA
[ ] Step-by-step PCA manual
[ ] PCA dengan scikit-learn

EVALUASI
[ ] Explained Variance Ratio
[ ] Scree Plot

VISUALISASI
[ ] 2D PCA Plot
[ ] Biplot
```

---

## 🔗 Terhubung ke

- [[MOC — Machine Learning]] — parent MOC
- [[IMV — Tugas PCA Individu]] — tugas yang menggunakan materi ini
- [[MOC — Statistika]] — fondasi statistik yang relevan

---

_Note ini adalah living document 