import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================
# 1. Baca citra
# =====================
img = cv2.imread("image2.jpeg")

# Cek jika gambar gagal dibaca
if img is None:
    print("Gambar tidak ditemukan. Pastikan file image2.jpeg ada di folder yang sama.")
    exit()

# Konversi BGR → RGB (agar warna benar di matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =====================
# 2. Representasi matriks & vektor
# =====================
print("Ukuran citra (tinggi, lebar):", gray.shape)

matrix_5x5 = gray[:5, :5]
print("\nMatriks 5x5 pertama:\n", matrix_5x5)

vector = gray.flatten()
print("\n10 elemen pertama vektor:", vector[:10])
print("Panjang vektor:", len(vector))

# =====================
# 3. Analisis parameter citra
# =====================
height, width = gray.shape

resolution = width * height
bit_depth = 8
intensity_levels = 2 ** bit_depth
aspect_ratio = width / height

memory_bits = resolution * bit_depth
memory_bytes = memory_bits / 8
memory_mb = memory_bytes / (1024 * 1024)

print("\nResolusi:", width, "x", height)
print("Total piksel:", resolution)
print("Bit depth:", bit_depth)
print("Jumlah intensitas:", intensity_levels)
print("Aspect ratio:", aspect_ratio)
print("Ukuran memori (MB):", memory_mb)

# Resolusi 2x dan bit depth setengah
new_pixels = resolution * 4
new_bit_depth = bit_depth / 2

new_bits = new_pixels * new_bit_depth
new_bytes = new_bits / 8
new_mb = new_bytes / (1024 * 1024)

print("\nMemori jika resolusi 2x & bit depth 1/2 (MB):", new_mb)

# =====================
# 4. Manipulasi dasar
# =====================
crop = img_rgb[400:1100, 200:700]
resize = cv2.resize(img_rgb, (450, 800))
rotate90 = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
flip = cv2.flip(img_rgb, 1)

# =====================
# 5. Tampilkan hasil (MATPLOTLIB)
# =====================
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(crop)
plt.title("Crop")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(resize)
plt.title("Resize")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(rotate90)
plt.title("Rotasi 90°")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(flip)
plt.title("Flip Horizontal")
plt.axis("off")

plt.tight_layout()
plt.show()