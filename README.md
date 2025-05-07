# 🍎🍌🍓 Klasifikasi Gambar Buah dengan CNN dan MobileNetV2

Proyek ini bertujuan untuk mengklasifikasikan gambar buah menjadi 5 kelas: Apel, Pisang, Mangga, Jeruk, dan Stroberi 🍎🍌🥭🍊🍓  
Model dikembangkan menggunakan TensorFlow, Keras, dan MobileNetV2 dengan pendekatan fine-tuning dan augmentasi data. Model ini dilatih untuk mengenali berbagai jenis buah dan diekspor ke format SavedModel, TensorFlow Lite (TFLite), dan TensorFlow.js (TFJS).

---

### 1. Sumber Dataset

Dataset terhubung langsung melalui Kaggle API menggunakan `opendatasets`.  
Pastikan sudah memiliki file `kaggle.json` yang valid untuk dapat mengakses dataset secara otomatis.

---

### 2. Arsitektur Model

Model dibangun menggunakan Keras `Sequential` API, dengan kombinasi:

- Pre-trained MobileNetV2 sebagai feature extractor
- Tambahan 1 layer `Conv2D` dan `MaxPooling2D`
- `GlobalAveragePooling2D` alih-alih `Flatten` untuk menghindari error dimensi
- `Dropout` untuk menghindari overfitting
- `Dense` layer sebagai classifier akhir

---

### 3. Training

Model dilatih menggunakan:

- `ImageDataGenerator` untuk augmentasi pada data latih
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Callback: `ModelCheckpoint` dan `EarlyStopping` kustom

---

### 4. Evaluasi

Model dievaluasi menggunakan:

- Confusion Matrix (dengan seaborn heatmap)
- Classification Report (precision, recall, f1-score)

---

### 5. Inference

Model dapat digunakan untuk memprediksi kelas gambar tunggal berukuran 150x150 piksel.
Contoh kode:

```python
from tensorflow.keras.preprocessing import image

img_path = 'contoh.jpeg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

model = tf.keras.models.load_model('best_model.keras')
pred = model.predict(img_array)
predicted_class = np.argmax(pred[0])
```

---

### 6. Export Model

Model disimpan dalam berbagai format:

- `.keras`
- SavedModel (untuk inference lebih lanjut)
- Dapat dikonversi ke TFLite atau TensorFlow.js

---

### 7. label.txt

Berikut daftar label kelas dalam urutan yang digunakan:

```
Apple
Banana
Mango
Orange
Strawberry
```

---

## 📝 Penulis

- Nama: _[Nabila Neva Rahmawati]_
- Email: _[nabilanevaaa@gmail.com]_
- ID Dicoding: _[nabilaneva]_
