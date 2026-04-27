# 🍊 UTS Machine Learning: Klasifikasi Oranges vs Grapefruit
**Metodologi:** Cross-Industry Standard Process for Data Mining (CRISP-DM)

Repositori ini berisi implementasi model *Machine Learning* untuk mengklasifikasikan buah jeruk (*orange*) dan jeruk bali (*grapefruit*) berdasarkan fitur fisik menggunakan perbandingan tiga algoritma: Decision Tree, Naive Bayes, dan Support Vector Machine (SVM).

---

## 1️⃣ Business Understanding
**Tujuan Bisnis**
Tujuan utama proyek ini adalah membangun model klasifikasi otomatis yang mampu membedakan buah jeruk (*orange*) dari jeruk bali (*grapefruit*) berdasarkan atribut fisik yang terukur. Dalam konteks dunia nyata, model semacam ini bisa diterapkan di lini sortir buah pada industri pertanian atau distribusi pangan — menggantikan proses seleksi manual yang lambat dan rawan kesalahan.

**Tujuan Machine Learning**
Secara teknis, ini adalah masalah *binary classification*: model menerima fitur numerik dari sebuah buah dan memprediksi labelnya (*orange* atau *grapefruit*). Keberhasilan diukur dari akurasi dan kemampuan generalisasi model pada data baru.

---

## 2️⃣ Data Understanding
Dataset yang digunakan mengukur karakteristik fisik buah dan memiliki 6 kolom utama.

**Struktur Dataset**
| Kolom | Tipe | Deskripsi |
| :--- | :--- | :--- |
| `name` | Kategorikal | Label target: *orange* atau *grapefruit* |
| `diameter` | Numerik (kontinu) | Diameter buah dalam cm |
| `weight` | Numerik (kontinu) | Berat buah dalam gram |
| `red` | Numerik (kontinu) | Nilai *channel* merah (RGB) dari warna kulit buah |
| `green` | Numerik (kontinu) | Nilai *channel* hijau (RGB) |
| `blue` | Numerik (kontinu) | Nilai *channel* biru (RGB) |

**Karakteristik Penting & Implikasi untuk Model**
* **Fitur prediktor:** 5 fitur numerik (`diameter`, `weight`, `red`, `green`, `blue`) semuanya siap pakai.
* **Target variabel:** `name` (biner) perlu di-*encode* menjadi 0/1 untuk sebagian model.
* Tidak ada fitur kategorikal di sisi prediktor, yang sangat menguntungkan untuk algoritma SVM dan Naive Bayes.
* **Fitur warna (RGB)** menarik karena *grapefruit* cenderung lebih pucat/kuning dibanding *orange* yang lebih oranye-merah — berpotensi menjadi pembeda kuat (meski Naive Bayes mengasumsikan fitur ini independen).
* **Diameter & weight** *grapefruit* secara umum lebih besar dari *orange* — ini akan sangat relevan dan mudah dieksploitasi oleh *Decision Tree* yang bekerja dengan *threshold* batas (split sederhana).
* Karena variasi metrik, **SVM** akan membutuhkan normalisasi/standarisasi fitur agar berskala seragam sebelum tahap *training*.

---

## 3️⃣ Data Preparation
Data yang digunakan dalam membuat model klasifikasi ini diambil dari platform Kaggle, yang berisi pencatatan atribut fisik dari buah jeruk (*orange*) dan jeruk bali (*grapefruit*). Dataset tersebut dapat diakses pada tautan berikut: [Kaggle - Oranges vs Grapefruit](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit). 

Terdapat total 10.000 sampel data pada dataset ini. Data tersebut mempunyai dua kelas target pada kolom `name` yang selanjutnya dikonversi ke dalam bentuk numerik biner (0 untuk *orange* dan 1 untuk *grapefruit*). Untuk mempersiapkan pemodelan, 5 fitur prediktor utama (diameter, berat, serta nilai warna RGB) dipisahkan dari kolom target, lalu dataset dibagi menjadi 80% data latih (*training set*) dan 20% data uji (*testing set*). Mengingat ada algoritma yang sangat sensitif terhadap perbedaan rentang nilai seperti SVM, seluruh fitur numerik tersebut wajib dinormalisasi menggunakan `StandardScaler` agar memiliki skala metrik yang seragam. 

---

## 4️⃣ Modeling
Proses pemodelan dilakukan dengan membandingkan tiga algoritma klasifikasi:

### 4.1 Decision Tree Classifier
Model ini bekerja secara intuitif dengan membelah data berdasarkan aturan kondisional (seperti *threshold* diameter atau berat buah). 

```python
from sklearn.tree import DecisionTreeClassifier

# Inisialisasi dan melatih model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
```

### 4.2 Naive Bayes Classifier (Gaussian)
Model probabilitas yang didasarkan pada Teorema Bayes. Varian Gaussian dipilih karena sangat cocok untuk memproses kelima fitur fisik kita yang bertipe numerik dan kontinu.

```python
from sklearn.naive_bayes import GaussianNB

# Inisialisasi dan melatih model Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
```

### 4.3 Support Vector Machine (SVM)
Model ini bekerja dengan mencari batas pemisah (*hyperplane*) paling optimal di antara kelas orange dan grapefruit. Karena SVM mengandalkan perhitungan jarak antar titik data di ruang dimensi, kita wajib menggunakan data yang sudah melewati proses *scaling* (`X_train_scaled`).

```python
from sklearn.svm import SVC

# Inisialisasi dan melatih model SVM menggunakan kernel RBF
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
```

---

## 5️⃣ Evaluasi & Analisis
Setelah proses pelatihan selesai, ketiga model diuji menggunakan 20% data *testing* (berisi 2000 sampel buah yang belum pernah dilihat oleh model). Berdasarkan metrik evaluasi yang dihasilkan, berikut adalah komparasi kinerjanya:

* 🥇 **Decision Tree Classifier** (Akurasi: 94.00%)
* 🥈 **Support Vector Machine / SVM** (Akurasi: 93.70%)
* 🥉 **Naive Bayes - Gaussian** (Akurasi: 92.00%)

Selain unggul dari segi akurasi keseluruhan, metrik pendukung seperti *Precision*, *Recall*, dan *F1-Score* untuk ketiga model juga menunjukkan angka di atas 0.90 secara merata, menandakan tidak adanya indikasi *overfitting* yang parah atau bias kelas.

### 🔍 Analisis Kelebihan dan Kelemahan (Konteks Dataset)
Meskipun Decision Tree keluar sebagai pemenang, penting untuk memahami bahwa performa ini sangat dipengaruhi oleh karakteristik dataset kita (hanya memiliki 5 kolom fitur/dimensi rendah).

1. **Decision Tree:** * **Kelebihan:** Sangat diuntungkan karena jumlah fiturnya sedikit (hanya 5 dimensi) dan memiliki relasi hierarki yang logis (buah yang diameternya lebih besar hampir pasti lebih berat). Model ini memotong keputusan (*split*) dengan sangat cepat dan aturannya mudah dibaca manusia.
   * **Kelemahan:** Jika di masa depan sistem ini di-*upgrade* dengan menambahkan puluhan sensor baru, Decision Tree akan sangat rentan terhadap *overfitting* dan kinerjanya akan merosot tajam.
2. **Support Vector Machine (SVM):**
   * **Kelebihan:** Penggunaan kernel RBF sangat ahli dalam mencari batas pemisah (*hyperplane*) yang melengkung (non-linear) antar fitur warna dan ukuran. Jika data memiliki ratusan fitur kompleks, SVM adalah model yang paling "tahan banting" dan berpotensi besar mengalahkan Decision Tree.
   * **Kelemahan:** Membutuhkan tahap standardisasi (*scaling*) mutlak. Selain itu, SVM ibarat *black-box*; lebih sulit menjelaskan logika keputusannya secara visual.
3. **Naive Bayes (Gaussian):**
   * **Kelebihan:** Kecepatan komputasinya adalah yang paling juara karena hanya menghitung probabilitas matematis murni. Sangat cocok jika ditanamkan pada perangkat berspesifikasi rendah (*microcontroller*).
   * **Kelemahan:** Asumsi utamanya adalah bahwa setiap fitur berdiri sendiri (*independent*). Pada dataset ini, fitur RGB sebenarnya saling berkaitan membentuk persepsi warna. Asumsi "kepolosan" inilah yang membuat akurasinya tertinggal sedikit di posisi terakhir.

> 💡 **Kesimpulan Akhir:** Pendekatan metodologi CRISP-DM pada proyek ini berhasil dieksekusi dengan baik. Untuk diimplementasikan pada sistem penyortiran buah otomatis saat ini, algoritma Decision Tree adalah rekomendasi utama karena performanya paling akurat, efisien, dan keputusannya transparan. Namun, jika arsitektur perangkat keras industri kelak diperluas untuk mendeteksi ratusan parameter yang kompleks, model SVM harus menjadi pertimbangan utama penggantinya.

---

## 6️⃣ Dokumentasi & Deployment

### 6.1 Struktur File Script (main.ipynb)

```text
main.ipynb
  ├── Import libraries (pandas, numpy, sklearn, dll)
  ├── Load dataset
  ├── Phase 3: Data Preparation
  │   ├── EDA & missing values check
  │   ├── Encoding & feature separation
  │   ├── Train-test split
  │   └── Scaling/normalization
  ├── Phase 4: Modeling
  │   ├── Decision Tree training & prediction
  │   ├── Naive Bayes training & prediction
  │   └── SVM training & prediction
  └── Phase 5: Evaluation
      ├── Accuracy calculation
      ├── Classification report & Confusion matrix
      └── Comparative analysis
```

### 6.2 Petunjuk Penggunaan (How to Run)
Keseluruhan metodologi CRISP-DM proyek ini diimplementasikan menggunakan bahasa pemrograman Python di dalam ekosistem Jupyter Notebook. Untuk mengeksekusi, memvalidasi, atau mengembangkan ulang kode pada repositori ini, silakan ikuti panduan berikut:

1. Pastikan **Python 3** dan ekosistem **Jupyter** sudah terpasang di komputer Anda.
2. Lakukan instalasi pustaka (*library*) pendukung yang dibutuhkan dengan menjalankan perintah berikut di terminal:
   ```bash
   pip install pandas scikit-learn jupyter
   ```
3. Unduh dataset (`citrus.csv`) dan letakkan file tersebut di dalam folder direktori yang sama dengan file script Jupyter Anda.
4. Buka Command Prompt/Terminal, arahkan navigasi ke folder proyek, lalu ketik `jupyter notebook`.
5. Buka file ekstensi `.ipynb` atau eksekusi blok kode secara berurutan (*Run All*) mulai dari tahapan impor pustaka hingga bagian pencetakan komparasi hasil evaluasi.
```
