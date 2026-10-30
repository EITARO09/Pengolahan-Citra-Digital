import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Menyiapkan dan Membaca Gambar
img = cv2.imread("download.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tinggi, lebar = img.shape[:2]

# 2. Implementasi Translation
# Menggeser gambar 50 piksel ke sumbu X (kanan) dan 50 piksel ke sumbu Y (bawah)
M_translasi = np.float32([
    [1, 0, 50],
    [0, 1, 50]
])
translasi = cv2.warpAffine(img, M_translasi, (lebar, tinggi))

# 3. Implementasi Scaling (Faktor 0.5 dan 2)
skala_setengah = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
skala_dua = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# 4. Implementasi Rotation (45°, 90°, dan 180°)
pusat = (lebar // 2, tinggi // 2)

M_45 = cv2.getRotationMatrix2D(pusat, 45, 1.0)
rotasi_45 = cv2.warpAffine(img, M_45, (lebar, tinggi))

M_90 = cv2.getRotationMatrix2D(pusat, 90, 1.0)
rotasi_90 = cv2.warpAffine(img, M_90, (lebar, tinggi))

M_180 = cv2.getRotationMatrix2D(pusat, 180, 1.0)
rotasi_180 = cv2.warpAffine(img, M_180, (lebar, tinggi))

# 5. Implementasi Horizontal Flip dan Vertical Flip
flip_horizontal = cv2.flip(img, 1)
flip_vertical = cv2.flip(img, 0)

# 6. Menampilkan citra asli dan seluruh hasil transformasi
gambar_plot = [
    (img, "Citra Asli"),
    (translasi, "Translation"),
    (skala_setengah, "Scaling (0.5x)"),
    (skala_dua, "Scaling (2.0x)"),
    (rotasi_45, "Rotation 45°"),
    (rotasi_90, "Rotation 90°"),
    (rotasi_180, "Rotation 180°"),
    (flip_horizontal, "Horizontal Flip"),
    (flip_vertical, "Vertical Flip")
]

# PERBAIKAN TATA LETAK (LAYOUT)
# Menggunakan objek figure dan axes untuk kontrol ruang yang lebih baik
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.ravel() # Mengubah matriks 3x3 menjadi array 1D agar mudah diiterasi

for i, (citra, judul) in enumerate(gambar_plot):
    axes[i].imshow(citra)
    # Menambahkan pad=15 agar judul memiliki jarak aman dari gambar di atasnya
    axes[i].set_title(judul, fontsize=12, pad=15)
    axes[i].axis("off")

# Menambahkan jarak horizontal (wspace) dan vertikal (hspace) antar subplot
plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95, hspace=0.6, wspace=0.3)
plt.show()