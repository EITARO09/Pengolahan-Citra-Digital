import cv2
import matplotlib.pyplot as plt

img = cv2.imread("plat.jpg")
if img is None:
    raise FileNotFoundError("plat.jpg tidak ditemukan")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mean = cv2.blur(rgb, (3,3))
gauss = cv2.GaussianBlur(rgb, (3,3), 0)
median = cv2.medianBlur(rgb, 3)

fig, ax = plt.subplots(1, 4, figsize=(14,4))
for a, im, t in zip(ax, [rgb, mean, gauss, median], ["Asli", "Mean", "Gaussian", "Median"]):
    a.imshow(im)
    a.set_title(t)
    a.axis("off")

plt.tight_layout()
plt.show()