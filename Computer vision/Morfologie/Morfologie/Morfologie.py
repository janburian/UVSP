# Importy knihoven 
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

def dilate(image, element):
    """
        Binary dilation function

        @param image - input image
        @param element - element
    """
    result = np.zeros(image.shape, dtype=np.uint8)
    indexes = np.where(image == 1)
    elements = np.where(element == 1)
    for y,x in zip(indexes[0], indexes[1]):
        res0 = y-len(elements)//2 + elements[0]
        res1 = x-len(elements)//2 + elements[1]
        result[res0, res1] = 1
    return result

# Vytvori element urcite dimenze
def createElement(dimension):
    elem = np.zeros([dimension, dimension])
    elem[:,0] = 1
    elem[0,:] = 1
    elem[:, len(elem)-1] = 1
    elem[len(elem)-1, :] = 1
    elem[len(elem)//2, len(elem)//2] = 1

    return elem

# Nacteni obrazku
nazevObrazku = "Binary_coins.png"
img = io.imread(nazevObrazku, as_gray=True)

# Prevod na binarni matici
binaryImg = np.uint8(img > 0)
element = createElement(3)
print(element)

# Dilatace
img_dilate = dilate(binaryImg, element)
img_rozdil = img_dilate - img

f, axarr = plt.subplots(1,2)
axarr[0].imshow(img, cmap="gray")
axarr[0].set_title('Original picture')
axarr[1].imshow(img_rozdil, cmap = "gray")
axarr[1].set_title('Picture outline')
plt.show()


""" 
    # Testovaci cast kodu
    img_test = np.zeros([7,7])
    img_test[2:4,2:3] = 1
    print(img_test)

    elem = createElement(3)
    print(elem)
    test_dilate = dilate(img_test, elem)
    print()
    print(test_dilate)
"""


