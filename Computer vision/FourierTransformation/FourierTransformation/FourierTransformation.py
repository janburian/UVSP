# Importy knihoven 
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

# Metoda vytvarejici masku o velikosti obrazku
def createMask(imgShape):
    mask = np.ones([imgShape[0], imgShape[1]])
    mask[126:131, 0:85] = 0
    mask[172:255, 126:130] = 0

    mask[126:131, 170:255] = 0
    mask[0:85, 127:131] = 0

    return mask 

# Nacteni obrazku
nazevObrazku = "twigs.jpg"
img = io.imread(nazevObrazku, as_gray=True)

# Fourierova transformace
ft = np.fft.fft2(img) # 2D Fourier Transform (FT)
ftshift = np.fft.fftshift(ft) # Change quadrants of the FT (1<-->3, 2<-->4)
spectrum = 20*np.log(np.abs(ftshift)) # Spectrum of FT

# Maska
imgSize = img.shape
mask = createMask(imgSize)

product = (mask * ftshift)

ift = np.abs(np.fft.ifft2(product))

# Vizualizace
f, axarr = plt.subplots(1,3)
axarr[0].imshow(img, cmap="gray")
axarr[0].set_title('Original picture')
axarr[1].imshow(spectrum, cmap = "gray")
axarr[1].set_title('Spectrum')
axarr[2].imshow(ift, cmap = "gray")
axarr[2].set_title('Restored picture')
plt.show()
