import os
import numpy as np
import imageio.v3 as iio
from scipy import ndimage
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def pilih_file_gambar():
    """Membuka jendela File Explorer untuk memilih gambar secara langsung."""
    root = tk.Tk()
    root.withdraw()  # Sembunyikan jendela utama Tkinter
    root.attributes('-topmost', True) # Munculkan jendela di paling depan
    
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar untuk Diproses",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    return file_path

def ubah_ke_grayscale(img):
    """Mengubah citra RGB menjadi derajat keabuan (grayscale)."""
    if img.ndim == 3:
        # Konversi standar RGB ke Grayscale
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img.copy()
    return gray.astype(np.uint8)

def ekualisasi_histogram(img_gray):
    """Meratakan sebaran warna (kontras) menggunakan ekualisasi histogram."""
    # 1. Hitung histogram (frekuensi kemunculan tiap nilai warna 0-255)
    hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])
    
    # 2 & 3. Hitung persentase kumulatif (CDF)
    cdf = hist.cumsum()
    
    # 4. Skalakan CDF ke rentang warna 0-255
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # 5. Pemetaan nilai piksel baru
    img_hasil = cdf_final[img_gray]
    return img_hasil

def proses_penajaman_scipy(img_gray):
    """Menajamkan garis tepi objek menggunakan filter Laplacian dari SciPy."""
    kernel_penajam = np.array([[0,  1, 0],
                               [1, -4, 1],
                               [0,  1, 0]])
    garis_tepi = ndimage.convolve(img_gray.astype(float), kernel_penajam)
    foto_tajam = np.clip(img_gray.astype(float) - garis_tepi, 0, 255)
    return foto_tajam.astype(np.uint8)

def main():
    print("Silakan pilih file gambar dari jendela yang muncul...")
    path_gambar = pilih_file_gambar()
    
    if not path_gambar:
        print("Pemilihan file dibatalkan.")
        return

    print(f"Membaca file: {path_gambar}")
    img_raw = iio.imread(path_gambar)
    
    # Pemrosesan Citra
    foto_asli = ubah_ke_grayscale(img_raw)
    foto_kontras = ekualisasi_histogram(foto_asli)
    foto_tajam = proses_penajaman_scipy(foto_kontras)

    print("Menampilkan dan menyimpan hasil...")
    # Visualisasi Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Kolom 1: Citra Awal Grayscale
    axes[0, 0].imshow(foto_asli, cmap='gray')
    axes[0, 0].set_title('1. Citra Awal (Grayscale)')
    axes[0, 0].axis('off')
    axes[1, 0].hist(foto_asli.ravel(), 256, [0, 256], color='gray')
    axes[1, 0].set_title('Histogram Citra Awal')

    # Kolom 2: Hasil Ekualisasi Histogram
    axes[0, 1].imshow(foto_kontras, cmap='gray')
    axes[0, 1].set_title('2. Hasil Ekualisasi Histogram')
    axes[0, 1].axis('off')
    axes[1, 1].hist(foto_kontras.ravel(), 256, [0, 256], color='blue')
    axes[1, 1].set_title('Histogram Ter-ekualisasi')

    # Kolom 3: Hasil Penajaman Tepi (SciPy)
    axes[0, 2].imshow(foto_tajam, cmap='gray')
    axes[0, 2].set_title('3. Hasil Penajaman Tepi')
    axes[0, 2].axis('off')
    axes[1, 2].hist(foto_tajam.ravel(), 256, [0, 256], color='green')
    axes[1, 2].set_title('Histogram Hasil Penajaman')

    plt.tight_layout()
    
    # Simpan screenshot hasil otomatis
    file_output = 'screenshot_hasil_praktikum.png'
    plt.savefig(file_output, dpi=300)
    print(f"Hasil visualisasi berhasil disimpan sebagai '{file_output}'.")
    
    plt.show()

if __name__ == '__main__':
    main()