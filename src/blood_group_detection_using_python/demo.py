"""Demo file."""

import cv2
import matplotlib.pyplot as plt
import skimage.filters as skif
import skimage.io as ski

## INPUT IMAGE
img1 = ski.imread("./s.jpg")


##Color Plane Extraction: RGB Green Plane
_,img2,_ = cv2.split(img1)


##Auto Threshold: Clustering
t = skif.threshold_otsu(img2, nbins=256)
print(t)
# plt.imshow(g, cmap="gray")
