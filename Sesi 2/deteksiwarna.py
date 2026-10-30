import cv2
import matplotlib.pyplot as plt

# 1. Membaca citra dari file
img = cv2.imread('laut.jpeg')

# Mengecek apakah gambar berhasil dibaca
if img is None:
    print("Gambar tidak ditemukan. Pastikan file laut.jpeg berada di folder yang sama.")
    exit()

# 2. Mengubah BGR menjadi RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3. Mengubah citra RGB menjadi grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. Mengubah grayscale menjadi citra biner
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 5. Menampilkan informasi citra
tinggi, lebar, channel = img.shape

print("=== INFORMASI CITRA ===")
print(f"Nama file       : laut.jpeg")
print(f"Lebar           : {lebar} piksel")
print(f"Tinggi          : {tinggi} piksel")
print(f"Jumlah channel  : {channel}")
print("=======================")

# 6. Membuat tampilan 4 citra
plt.figure(figsize=(12, 8))

# Citra asli
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("1. Citra Asli")
plt.axis('off')

# Citra RGB
plt.subplot(2, 2, 2)
plt.imshow(img_rgb)
plt.title("2. Citra RGB")
plt.axis('off')

# Citra Grayscale
plt.subplot(2, 2, 3)
plt.imshow(gray, cmap='gray')
plt.title("3. Citra Grayscale")
plt.axis('off')

# Citra Biner
plt.subplot(2, 2, 4)
plt.imshow(binary, cmap='gray')
plt.title("4. Citra Biner")
plt.axis('off')

# Menampilkan semua hasil
plt.tight_layout()
plt.show()