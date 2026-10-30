import imageio as img
import numpy as np
import matplotlib.pyplot as plt

def hist_darken(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    cdf = hist.cumsum()
    
    # --- Perubahan Algoritma Disini ---
    # Langkah normalisasi asli tetap ada untuk menjaga rentang cdf.
    # Namun, target cdf dimodifikasi agar "ditekan" menggunakan faktor gamma.
    # Nilai gamma > 1.0 akan membuat gambar lebih gelap.
    gamma_factor = 2.0  
    
    # Rumus target CDF yang baru, yang diterapkan transformasi gamma:
    cdf_depressed = 255 * ( (cdf / cdf.max()) ** gamma_factor )
    
    # Gunakan cdf_depressed sebagai target interpolasi, bukan cdf_normal linier.
    image_darkened = np.interp(image.flatten(), bins[:-1], cdf_depressed)
    
    # --- Akhir Perubahan ---
    
    return image_darkened.reshape(image.shape).astype(np.uint8)

image = img.imread("image copy 3.png")  # Ganti dengan path gambar yang sesuai

# Panggil fungsi yang sudah diubah
result_dark = hist_darken(image)

hist_img, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
# Hitung histogram untuk hasil yang digelapkan
hist_result_dark, bins = np.histogram(result_dark.flatten(), bins=256, range=[0,256])

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(image)
plt.title("Gambar Asli")

plt.subplot(2,2,2)
plt.imshow(result_dark)
plt.title("Gambar Darkened")

plt.subplot(2,2,3)
plt.plot(hist_img)
plt.title("Histogram Gambar Asli")

plt.subplot(2,2,4)
plt.plot(hist_result_dark)
plt.title("Histogram Gambar Darkened")

plt.show()