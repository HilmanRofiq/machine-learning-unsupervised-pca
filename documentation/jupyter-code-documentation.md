# Dokumentasi Proyek Machine Learning
# Visualisasi Data Berdimensi Tinggi Menggunakan Principal Component Analysis (PCA)

## 1. Deskripsi Proyek

Proyek ini bertujuan untuk mengurangi dimensi data kompleks agar lebih mudah divisualisasikan dan dianalisis menggunakan metode **Principal Component Analysis (PCA)**.

PCA adalah salah satu teknik *dimensionality reduction* dalam kategori **unsupervised learning**, yang artinya algoritma ini belajar dari struktur internal data tanpa membutuhkan label. PCA bekerja dengan mencari arah-arah (principal components) di mana data paling bervariasi, lalu memproyeksikan data ke arah tersebut sehingga dimensi yang tinggi dapat dipadatkan menjadi 2 atau 3 dimensi yang bisa divisualisasikan.

Dataset yang digunakan adalah **Digits Dataset** dari `sklearn.datasets` yaitu kumpulan gambar angka tulisan tangan (0–9) yang masing-masing direpresentasikan sebagai vektor 64 dimensi (grid piksel 8×8).


## 2. Tujuan Proyek

- Memahami konsep dan cara kerja PCA sebagai metode reduksi dimensi
- Mengimplementasikan PCA pada data berdimensi tinggi (64 fitur)
- Memvisualisasikan hasil reduksi dimensi dalam 2D dan 3D
- Menganalisis kontribusi setiap fitur terhadap principal component
- Membuktikan efektivitas PCA melalui rekonstruksi gambar

## 3. Dataset

| Informasi | Keterangan |
|---|---|
| **Nama Dataset** | Digits Dataset |
| **Sumber Dataset** | `sklearn.datasets.load_digits()` |
| **Jumlah Sampel** | 1.797 gambar |
| **Jumlah Fitur** | 64 fitur (piksel 8×8) |
| **Kelas / Label** | 10 kelas (angka 0 hingga 9) |
| **Jenis Task** | Unsupervised Learning (Dimensionality Reduction) |
| **Tipe Data** | Numerik (nilai piksel: 0–16) |
| **Missing Value** | Tidak ada (0) |

### Penjelasan Fitur

Setiap sampel dalam dataset ini adalah gambar angka tulisan tangan berukuran **8×8 piksel**. Gambar tersebut diratakan (*flatten*) menjadi vektor 1 dimensi dengan 64 elemen — setiap elemen merepresentasikan nilai kecerahan satu piksel dalam rentang 0 (putih) hingga 16 (hitam).

```
Contoh representasi angka "0" dalam bentuk grid 8×8:

[ 0,  0,  5, 13,  9,  1,  0,  0]
[ 0,  0, 13, 15, 10, 15,  5,  0]
[ 0,  3, 15,  2,  0, 11,  8,  0]
[ 0,  4, 12,  0,  0,  8,  8,  0]
[ 0,  5,  8,  0,  0,  9,  8,  0]
[ 0,  4, 11,  0,  1, 12,  7,  0]
[ 0,  2, 14,  5, 10, 12,  0,  0]
[ 0,  0,  6, 13, 10,  0,  0,  0]

→ Diratakan menjadi vektor 64 elemen:
[0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, ...]
```

### Distribusi Kelas

| Kelas (Digit) | Jumlah Sampel |
|:---:|:---:|
| 0 | 178 |
| 1 | 182 |
| 2 | 177 |
| 3 | 183 |
| 4 | 181 |
| 5 | 182 |
| 6 | 181 |
| 7 | 179 |
| 8 | 174 |
| 9 | 180 |
| **Total** | **1.797** |

Dataset ini **sangat seimbang** — setiap kelas memiliki jumlah sampel yang hampir sama (174–183 sampel), sehingga tidak ada bias kelas yang perlu ditangani.

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Informasi Umum Dataset

```python
# Load dataset
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np

digits = load_digits()
X = digits.data    # shape: (1797, 64)
y = digits.target  # label: 0–9

# Buat DataFrame
df = pd.DataFrame(X, columns=[f'pixel_{i}' for i in range(64)])
df['label'] = y

print(df.shape)        # (1797, 65)
print(df.info())
print(df.describe())
```

**Output ringkasan:**

| Info | Nilai |
|---|---|
| Jumlah baris | 1.797 |
| Jumlah kolom | 64 (fitur) + 1 (label) |
| Tipe data | float64 |
| Missing value | 0 |
| Nilai minimum piksel | 0.0 |
| Nilai maksimum piksel | 16.0 |

### 4.2 Cek Missing Value

```python
print(df.isnull().sum().sum())  # Output: 0
```

Tidak ditemukan missing value pada dataset ini. Karena kebanyakan library dari scikitlearn rata rata sudah bersih.

### 4.3 Visualisasi Sample Gambar

Setiap baris data diubah kembali ke bentuk grid 8×8 untuk memvisualisasikan gambar aslinya:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 10, figsize=(18, 4))
for digit in range(10):
    idx = np.where(y == digit)[0][0]
    axes[0, digit].imshow(X[idx].reshape(8, 8), cmap='gray_r')
    axes[0, digit].set_title(f'Digit: {digit}', fontsize=9)
    axes[0, digit].axis('off')
plt.show()
```
![image](https://hackmd.io/_uploads/BJUZtaYefl.png)

> Setiap gambar berukuran 8×8 = **64 piksel** yang akan menjadi 64 fitur dalam proses PCA.

### 4.4 Distribusi Kelas

```python
import seaborn as sns

sns.countplot(x=y)
plt.title('Distribusi Kelas — Digits Dataset')
plt.xlabel('Digit')
plt.ylabel('Jumlah Sampel')
plt.show()
```

Dataset terdistribusi merata. Setiap digit memiliki sekitar 178–183 sampel.

### 4.5 Korelasi Antar Fitur

```python
corr = df.iloc[:, :20].corr()
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Matrix — 20 Fitur Pertama')
plt.show()
```
![image](https://hackmd.io/_uploads/Sy5HtaKlGx.png)

Terlihat banyak piksel yang berkorelasi tinggi satu sama lain, terutama piksel-piksel yang berdekatan secara spasial. Ini adalah **indikasi kuat** bahwa PCA akan efektif memadatkan informasi karena banyak fitur yang redundan.

## 5. Preprocessing Data

### 5.1 Mengapa Perlu Preprocessing?

PCA sensitif terhadap skala fitur. Fitur dengan nilai besar akan mendominasi principal component jika tidak dinormalisasi terlebih dahulu. Oleh karena itu, **StandardScaler wajib diterapkan** sebelum PCA.

### 5.2 Standardisasi dengan StandardScaler

StandardScaler mengubah setiap fitur agar memiliki **mean = 0** dan **standar deviasi = 1**.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Perbandingan sebelum dan sesudah scaling (fitur ketiga):**

| Statistik | Sebelum Scaling | Sesudah Scaling |
|---|:---:|:---:|
| Mean | 5.2048 | ≈ 0.000000 |
| Std | 4.7535 | 1.000000 |
| Min | 0.0 | -1.0949 |
| Max | 16.0 | 2.2710 |

> **Catatan:** Pada dataset Digits, semua piksel sudah berada dalam rentang 0–16 sehingga efek scaling tidak sedramatis dataset lain. Namun standardisasi tetap diterapkan sebagai *best practice* dalam implementasi PCA.


## 6. Metode yang Digunakan

### Principal Component Analysis (PCA)

PCA adalah metode **unsupervised learning** untuk reduksi dimensi yang bekerja dengan cara:

1. Menghitung Covariance Matrix dari data yang sudah di-scale
2. Menghitung Eigenvalue dan Eigenvector dari covariance matrix
3. Mengurutkan eigenvector berdasarkan eigenvalue (dari besar ke kecil)
4. Memilih sejumlah `n` eigenvector teratas sebagai principal components
5. Memproyeksikan data asli ke ruang baru yang dibentuk oleh principal components tersebut

**Alasan memilih PCA:**

- Data Digits memiliki 64 dimensi yang mustahil divisualisasikan secara langsung
- Banyak piksel yang saling berkorelasi → informasi redundan dapat dipadatkan
- PCA dapat memvisualisasikan struktur data dan memperlihatkan apakah kelas-kelas yang berbeda (angka 0–9) terpisah secara alami dalam ruang fitur
- PCA bersifat deterministik dan hasilnya dapat diinterpretasikan secara matematis


## 7. Proses Training / Implementasi PCA

### 7.1 Menentukan Jumlah Komponen Optimal (Scree Plot)

Sebelum memilih jumlah komponen, kita jalankan PCA dengan semua komponen untuk melihat distribusi variance:

```python
from sklearn.decomposition import PCA
import numpy as np

# PCA dengan semua komponen
pca_full = PCA()
pca_full.fit(X_scaled)

evr  = pca_full.explained_variance_ratio_
cumr = np.cumsum(evr)

# Berapa PC untuk mencapai threshold tertentu?
n_90 = np.argmax(cumr >= 0.90) + 1   # hasil: 31 komponen
n_95 = np.argmax(cumr >= 0.95) + 1   # hasil: 40 komponen
```
![image](https://hackmd.io/_uploads/Bkn15TFlzg.png)

**Hasil:**

| Threshold Variance | Jumlah PC yang Dibutuhkan |
|:---:|:---:|
| 90% | 31 PC |
| 95% | 40 PC |
| 100% | 64 PC (data asli) |
| Visualisasi 2D | 2 PC |
| Visualisasi 3D | 3 PC |

> Untuk keperluan visualisasi, kita cukup menggunakan 2–3 PC. Meskipun variance yang dipertahankan lebih kecil, tujuan utamanya adalah melihat pola dan struktur data — bukan rekonstruksi yang sempurna.

### 7.2 Fit dan Transform PCA

```python
# PCA 2D — untuk visualisasi scatter plot
pca_2d = PCA(n_components=2, random_state=42)
X_2d   = pca_2d.fit_transform(X_scaled)

# PCA 3D — untuk visualisasi 3D interaktif
pca_3d = PCA(n_components=3, random_state=42)
X_3d   = pca_3d.fit_transform(X_scaled)

print(f"Shape asal    : {X_scaled.shape}")   # (1797, 64)
print(f"Shape 2D      : {X_2d.shape}")       # (1797, 2)
print(f"Shape 3D      : {X_3d.shape}")       # (1797, 3)
```

**Reduksi dimensi yang berhasil dilakukan:**

| | Shape | Dimensi |
|---|:---:|:---:|
| Data asli | (1797, **64**) | 64 dimensi |
| Setelah PCA 2D | (1797, **2**) | 2 dimensi |
| Setelah PCA 3D | (1797, **3**) | 3 dimensi |


## 8. Evaluasi Model

Evaluasi PCA tidak menggunakan metrik seperti akurasi (karena tidak ada label prediksi), melainkan menggunakan **Explained Variance Ratio** yang mengukur seberapa banyak informasi dari data asli yang berhasil dipertahankan.

### 8.1 Explained Variance Ratio

```python
print(f"PC1 : {pca_2d.explained_variance_ratio_[0]*100:.2f}%")
print(f"PC2 : {pca_2d.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total (PC1+PC2) : {pca_2d.explained_variance_ratio_.sum()*100:.2f}%")
print(f"Total (PC1+PC2+PC3) : {pca_3d.explained_variance_ratio_.sum()*100:.2f}%")
```

**Hasil Evaluasi:**

| Metrik | Nilai |
|---|:---:|
| Variance PC1 | 12.03% |
| Variance PC2 | 9.56% |
| Total variance (2D) | 21.59% |
| Total variance (3D) | 30.04% |
| PC yang dibutuhkan untuk 90% variance | 31 PC |
| PC yang dibutuhkan untuk 95% variance | 40 PC |

> **Catatan interpretasi:** Total variance 21.59% untuk 2D terlihat kecil, namun ini wajar untuk dataset dengan 64 dimensi. Tujuan PCA 2D di sini adalah **visualisasi pola dan struktur cluster**, bukan rekonstruksi data yang sempurna. Cluster-cluster angka 0–9 tetap terlihat terpisah secara visual meski variance yang dipertahankan tidak besar.

### 8.2 Rekonstruksi Gambar sebagai Bukti Efektivitas
 
Salah satu cara paling intuitif untuk mengevaluasi PCA adalah dengan merekonstruksi data kembali ke dimensi aslinya menggunakan `inverse_transform`:
 
```python
n_components_list = [1, 2, 5, 10, 20, 30, 40, 64]
 
for n in n_components_list:
    pca_n = PCA(n_components=n, random_state=42)
    X_reduced       = pca_n.fit_transform(X_scaled)
    X_reconstructed = pca_n.inverse_transform(X_reduced)
    var = pca_n.explained_variance_ratio_.sum() * 100
    print(f"n={n:2d} PC → {var:.2f}% variance dipertahankan")
```
 
**Hasil rekonstruksi:**
 
| Jumlah PC | Variance Dipertahankan | Kualitas Gambar |
|:---:|:---:|---|
| 1 | 12.03% | Sangat buram, tidak bisa dikenali |
| 2 | 21.59% | Buram, samar-samar terlihat bentuknya |
| 5 | 41.40% | Mulai terlihat, masih kasar |
| 10 | 58.87% | Cukup jelas, angka bisa dikenali |
| 20 | 79.31% | Jelas, detail mulai muncul |
| 30 | 89.32% | Hampir sempurna |
| 40 | 95.08% | Sangat mirip dengan aslinya |
| 64 | 100.00% | Identik dengan gambar asli |


## 9. Hasil Output

### 9.1 Visualisasi 2D PCA Plot

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 7))
for digit in range(10):
    mask = y == digit
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               label=str(digit), alpha=0.7, s=18)

ax.set_xlabel(f'PC1 (12.03% variance)')
ax.set_ylabel(f'PC2 (9.56% variance)')
ax.set_title('PCA 2D — Digits Dataset')
ax.legend(title='Digit')
plt.show()
```

**Interpretasi hasil:**
- Setiap titik mewakili satu gambar angka, diwarnai berdasarkan kelasnya
- Titik-titik dengan warna yang sama (angka yang sama) cenderung berkumpul di area yang berdekatan
- Angka yang bentuknya mirip secara visual (seperti 3, 8, 9) posisinya berdekatan dalam plot
- Angka yang bentuknya berbeda jauh (seperti 0 vs 1) terpisah dengan jelas
- Ini membuktikan bahwa PCA berhasil mempertahankan struktur kemiripan antar data
![pca_2d](https://hackmd.io/_uploads/rJYqLaYlfg.png)

### 9.2 Visualisasi 3D PCA Plot (Interaktif)

```python
import plotly.express as px
import pandas as pd

df_3d = pd.DataFrame(X_3d, columns=['PC1', 'PC2', 'PC3'])
df_3d['Digit'] = y.astype(str)

fig = px.scatter_3d(df_3d, x='PC1', y='PC2', z='PC3',
                    color='Digit', opacity=0.75,
                    title='PCA 3D — Digits Dataset')
fig.update_traces(marker=dict(size=3))
fig.show()
```
![image](https://hackmd.io/_uploads/ByHeDaFxMx.png)

Plot 3D menampilkan pemisahan cluster yang lebih jelas dibanding 2D karena menangkap 30.04% variance (lebih banyak 8.45% dari versi 2D). Plot ini bersifat interaktif — bisa diputar dan di-zoom langsung di notebook.

### 9.3 Rekonstruksi Gambar

```
Contoh rekonstruksi digit "3" dengan berbagai jumlah PC:

n=1  PC  → [gambar sangat buram]   ~12% info
n=10 PC  → [gambar mulai dikenali] ~51% info
n=30 PC  → [gambar hampir sempurna] ~81% info
n=64 PC  → [gambar identik asli]   100% info

Kesimpulan: Hanya dengan 10 dari 64 komponen (15.6% dari total),
angka sudah dapat dikenali — membuktikan efisiensi kompresi PCA.
```
![image (1)](https://hackmd.io/_uploads/SkaVP6Kgfe.png)
### 9.4 Hasil Biplot

Biplot menampilkan dua informasi sekaligus:
- **Titik berwarna** = posisi setiap sampel dalam ruang PC1–PC2
- **Panah merah** = arah kontribusi 8 fitur (piksel) terpenting

Piksel yang letaknya di bagian tengah gambar (area aktif penulisan angka) cenderung memiliki loading yang tinggi dan berpengaruh besar terhadap principal component.
![image](https://hackmd.io/_uploads/rke9D6KxGe.png)


## 10. Kesimpulan

Berdasarkan hasil implementasi dan analisis PCA pada Digits Dataset, dapat disimpulkan:

1. **PCA berhasil mereduksi dimensi** dari 64 dimensi menjadi 2 dimensi, sambil mempertahankan 21.59% variance. Meskipun angka ini terlihat kecil, struktur cluster 10 kelas angka tetap terlihat jelas secara visual.

2. **Efisiensi kompresi terbukti nyata** — hanya dengan 10 principal component (15.6% dari total 64), gambar angka sudah dapat dikenali kembali saat direkonstruksi. Ini menunjukkan bahwa sebagian besar informasi visual memang tersimpan di sedikit komponen utama.

3. **Untuk mempertahankan 90% informasi** dibutuhkan 31 PC, dan untuk 95% dibutuhkan 40 PC — jauh lebih efisien dibanding menyimpan seluruh 64 dimensi.

4. **StandardScaler terbukti penting** sebagai langkah preprocessing wajib sebelum PCA agar setiap fitur mendapat perlakuan yang setara dalam penghitungan variance.

## 11. Kendala dan Solusi

| Kendala | Solusi |
|---|---|
| Data berdimensi tinggi (64 fitur) tidak dapat divisualisasikan secara langsung | Diterapkan PCA untuk mereduksi ke 2D dan 3D |
| PCA hanya menangkap 21.59% variance pada 2D | Ditambahkan visualisasi 3D (30.04%) dan analisis rekonstruksi gambar sebagai bukti efektivitas |
| Hasil PCA sulit diinterpretasikan tanpa konteks | Ditambahkan biplot dan loading matrix untuk menjelaskan kontribusi tiap fitur |

## 12. Link Kodingan

[Repository GitHub](https://github.com/HilmanRofiq/machine-learning-unsupervised-pca)


## 13. Link Presentasi dan Demo

[Video Presentasi + Demo Notebook](https://drive.google.com/drive/folders/17gsgNHOCBO0UsBRtw2-XDUayqvkgyOzD?usp=sharing)

---