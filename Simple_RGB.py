from PIL import Image
from sklearn import decomposition
from sklearn.datasets import load_sample_images
from skimage.measure._structural_similarity import compare_ssim
from skimage import measure
from skimage import metrics
import numpy as np
from matplotlib import pyplot as plt
import os


def normalization(array):
    array = np.around(array, 0)
    min_ele = array.min(initial=None)
    array -= min_ele
    max_ele = array.max(initial=None)
    array = 255 * array / max_ele
    return np.around(array, 0)


Max_conversion_rate = 80  # Set number -> 0-100 [%] of the original picture size.
Normalization = 1  # Normalization of results / Set 0/1
Picture_choice = 1  # Selection photo / Set 0/1


Picture = "Base_RGB.jpg"
Picture_output = "RGB_output.jpg"
Picture_compressed = "RGB_compressed_output.jpg"
space = ' '
database = load_sample_images()
Base_img = database.images[Picture_choice]

Show_img = Image.fromarray(Base_img)
Show_img.save(Picture)

Height, Width = Base_img.shape[0], Base_img.shape[1]

f_original = open(Picture)
f_original.seek(0, os.SEEK_END)
Base_weight = f_original.tell()
f_original.close()

Range = int(Max_conversion_rate * Height / 100) if int(Max_conversion_rate * Height / 100) < Width else Width - 1 if Height == Width else Width

RGB_Encoded_array = []
RGB_Compressed_array = []

svd = decomposition.TruncatedSVD(n_components=Range if Range != 0 else 1)
for RGB_each_array in range(3):
    RGB = Base_img[:, :, RGB_each_array]
    RGB_simple_array = svd.fit_transform(RGB)
    RGB_Compressed_array.append(RGB_simple_array)
    RGB_Encoded_array.append((svd.inverse_transform(RGB_simple_array)))


if Normalization == 1:
    Image_encoded = normalization(np.dstack(RGB_Encoded_array))
    Image_coded = normalization(np.dstack(RGB_Compressed_array))
    RGB_Encoded_array = normalization(RGB_Encoded_array)
else:
    Image_encoded = np.dstack(RGB_Encoded_array)
    Image_coded = np.dstack(RGB_Compressed_array)
    RGB_Encoded_array = RGB_Encoded_array

vr_score = svd.explained_variance_ratio_.sum()
psnr_score = metrics.peak_signal_noise_ratio(Base_img, Image_encoded, data_range=255)
mse_score = measure.simple_metrics.mean_squared_error(Base_img, Image_encoded)
ssim_score = compare_ssim(Base_img, Image_encoded, data_range=255, multichannel=True)

# Zapisanany obraz z rozkompresowanej macierzy / porownanie rozmiaru do originalu
new_im_1 = Image.fromarray(Image_encoded.astype(np.uint8), mode='RGB')
new_im_1.save(Picture_output)
fo = open(Picture_output)
fo.seek(0, os.SEEK_END)
Encode_Weigth = fo.tell()
fo.close()
size = 100 * Encode_Weigth / Base_weight

# Zapisana skompresowana macierz SVD / odtworzona w jpg i sprawdzanie jej rozmiaru
new_im_2 = Image.fromarray(Image_coded.astype(np.uint8))
new_im_2.save(Picture_compressed)
fo = open(Picture_compressed)
Height_Coded, Width_Coded = Image_coded.shape[0], Image_coded.shape[1]
fo.seek(0, os.SEEK_END)
Code_Weigth = fo.tell()
fo.close()
size_compress = 100 * Code_Weigth / Base_weight

Title = "PSNR:" + str(round(psnr_score, 3)) + "[dB]" + 10*space + "SSIM:" + str(round(ssim_score, 3))+"\n" + \
        + 14 * space + "MSE:" + str(round(mse_score, 3)) + 10*space + "Variance ratio:" + str(round(vr_score, 3))

fig = plt.figure(1)
fig.suptitle(Title)

plt.subplot(2, 2, 3)
plt.imshow(Image_encoded/255)
plt.title("Encoded photo")
plt.xlabel("Weigth_Ratio =" + str(round(size, 3))+"%" + "    Weigth =" + str(Encode_Weigth) + " bytes")
plt.ylabel("Size: " + str(Width) + "X" + str(Height))
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 1)
plt.imshow(Base_img)
plt.title("Original photo")
plt.xlabel("Weigth_Ratio = 100%   Weigth =" + str(Base_weight) + " bytes")
plt.ylabel("Size: " + str(Width) + "X" + str(Height))
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 3)
plt.imshow(Image_coded/255)
plt.title("Visualization of the compressed photo")
plt.xlabel("Weigth_Ratio =" + str(round(size_compress, 3))+"%" + "    Weigth =" + str(Code_Weigth) + " bytes")
plt.ylabel("Size: " + str(Width_Coded) + "X" + str(Height_Coded))
plt.xticks([])
plt.yticks([])

fig = plt.figure(2)
fig.suptitle("RGB distribution")

plt.subplot(2, 2, 1)
plt.imshow(Image_encoded/255)
plt.ylabel("Size: " + str(Width) + "X" + str(Height))
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(RGB_Encoded_array[0], cmap='Reds')
plt.ylabel("Size: " + str(Width) + "X" + str(Height))
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(RGB_Encoded_array[1], cmap='Greens')
plt.ylabel("Size: " + str(Width) + "X" + str(Height))
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(RGB_Encoded_array[2], cmap='Blues')
plt.ylabel("Size: " + str(Width) + "X" + str(Height))
plt.xticks([])
plt.yticks([])
plt.show()
