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
    array=array.astype(np.float64)
    min_ele = array.min(initial=None)
    array -= min_ele
    max_ele = array.max(initial=None)
    array = 255 * array / max_ele
    return array.astype(np.uint8)

######
Max_conversion_rate = 86  # set number -> 0-100 [%] !!!!!!!!!!!
Transpone = 0  # set 0/1
######

Picture = "Base_Mono.jpg"
Picture_output = "Mono_output.jpg"
Picture_compressed = "Mono_compressed_output.jpg"
space = ' '
dataset = load_sample_images()
first_img = dataset.images[1]

Show_img = Image.fromarray(first_img).convert('L')
Show_img.save(Picture)

Base_img = np.array(Image.open(Picture))

Base_img = np.transpose(Base_img) if Transpone == 1 else Base_img
Height, Width = Base_img.shape[0], Base_img.shape[1]

f_original = open(Picture)
f_original.seek(0, os.SEEK_END)
Base_weight = f_original.tell()
f_original.close()
Range = int(Max_conversion_rate * (Height-1) / 100) if int(Max_conversion_rate * (Height-1) / 100) < Width else Width - 1

svd = decomposition.TruncatedSVD(n_components=Range if Range != 0 else 1)
Image_coded = svd.fit_transform(Base_img)
svd_score = svd.explained_variance_ratio_.sum()

Image_encoded = svd.inverse_transform(Image_coded)

psnr_score = metrics.peak_signal_noise_ratio(Base_img, Image_encoded, data_range=255)
mse_score = measure.simple_metrics.mean_squared_error(Base_img, Image_encoded)
ssim_score = compare_ssim(Base_img, Image_encoded, data_range=255)

# Zapisanany obraz z rozkompresowanej macierzy / porownanie rozmiaru do originalu
new_im_1 = Image.fromarray(Image_encoded)
new_im_1 = new_im_1.convert("L")
new_im_1.save(Picture_output)
fo = open(Picture_output)
fo.seek(0, os.SEEK_END)
Encode_Weigth = fo.tell()
fo.close()
size = (100 * Encode_Weigth / Base_weight)


# Zapisana skompresowana macierz SVD / odtworzona w jpg i sprawdzanie jej rozmiaru
new_im_2 = Image.fromarray(Image_coded)
new_im_2 = new_im_2.convert("L")
new_im_2.save(Picture_compressed)
fo = open(Picture_compressed)
Height_Coded, Width_Coded = Image_coded.shape[0], Image_coded.shape[1]
fo.seek(0, os.SEEK_END)
Code_Weigth = fo.tell()
fo.close()
size_compress = 100 * Code_Weigth / Base_weight

Title = "PSNR:" + str(round(psnr_score, 3)) + "[dB]" + 10*space + "SSIM:" + str(round(ssim_score, 3))+"\n" + \
        + 14 * space + "MSE:" + str(round(mse_score, 3)) + 10*space + "Variance ratio:" + str(round(svd_score, 3))
fig = plt.figure(1)
fig.suptitle(Title)
plt.subplot(2, 2, 3)
plt.title("Encoded photo")
plt.imshow(Image_encoded, cmap='gray', vmin=0, vmax=255)
plt.xlabel("Weigth_Ratio =" + str(round(size, 3))+"%" + "    Weigth =" + str(Encode_Weigth) + " bytes")
plt.ylabel("Size: " + str(Width) + "X" + str(Height))
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 1)
plt.title("Original photo")
plt.imshow(Base_img, cmap='gray', vmin=0, vmax=255)
plt.xlabel("Weigth_Ratio = 100%   Weigth =" + str(Base_weight) + " bytes")
plt.ylabel("Size: " + str(Width) + "X" + str(Height))
plt.xticks([])
plt.yticks([])
plt.subplot(1, 3, 3)
plt.title("Visualization of the compressed photo")
plt.imshow(Image_coded, cmap='gray', vmin=0, vmax=255)
plt.xlabel("Weigth_Ratio =" + str(round(size_compress, 3))+"%" + "    Weigth =" + str(Code_Weigth) + " bytes")
plt.ylabel("Size: " + str(Width_Coded) + "X" + str(Height_Coded))
plt.xticks([])
plt.yticks([])
plt.show()
