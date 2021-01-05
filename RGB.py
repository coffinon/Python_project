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
    array = np.around(array, 4)
    min_ele = array.min(initial=None)
    array -= min_ele
    max_ele = array.max(initial=None)
    array = 255 * array / max_ele
    return np.around(array, 4)


Max_conversion_rate = 100  # Set number -> 0-100 [%] of the original picture size.
Normalization = 1  # Normalization of results /  Set 0/1
Picture_choice = 0  # Selection photo / Set 0/1


Picture = "Base_RGB.jpg"
Picture_output = "RGB_output.jpg"
Picture_compressed = "RGB_compressed_output.jpg"
space = ' '
mse_score, ssim_score, psnr_score, vr_score, size, size_compress = [], [], [], [], [], []
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

Range = 2 if Range == 0 else Range

for n in range(1, Range):
    svd = decomposition.TruncatedSVD(n_components=n if Range != 0 else 1)
    RGB_Encoded_array = []
    RGB_Compressed_array = []
    for RGB_each_array in range(3):
        RGB = Base_img[:, :, RGB_each_array]
        RGB_simple_array = svd.fit_transform(RGB)
        RGB_Compressed_array.append(RGB_simple_array)
        RGB_Encoded_array.append((svd.inverse_transform(RGB_simple_array)))

    if Normalization == 1:
        Image_encoded = normalization(np.dstack(RGB_Encoded_array))
        Image_coded = normalization(np.dstack(RGB_Compressed_array))
    else:
        Image_encoded = np.dstack(RGB_Encoded_array)
        Image_coded = np.dstack(RGB_Compressed_array)

    vr_score.append(svd.explained_variance_ratio_.sum())
    psnr_score.append(metrics.peak_signal_noise_ratio(Base_img, Image_encoded, data_range=255))
    mse_score.append(measure.simple_metrics.mean_squared_error(Base_img, Image_encoded))
    ssim_score.append(compare_ssim(Base_img, Image_encoded, data_range=255, multichannel=True))

    # Zapisanany obraz z rozkompresowanej macierzy / porownanie rozmiaru do originalu
    new_im_1 = Image.fromarray(Image_encoded.astype(np.uint8), mode='RGB')
    new_im_1.save(Picture_output)
    fo = open(Picture_output)
    fo.seek(0, os.SEEK_END)
    Encode_Weigth = fo.tell()
    fo.close()
    size.append(100 * Encode_Weigth / Base_weight)

    # Zapisana skompresowana macierz SVD / odtworzona w jpg i sprawdzanie jej rozmiaru
    new_im_2 = Image.fromarray(Image_coded.astype(np.uint8), mode='RGB')
    new_im_2.save(Picture_compressed)
    fo = open(Picture_compressed)
    Height_Coded, Width_Coded = Image_coded.shape[0], Image_coded.shape[1]
    fo.seek(0, os.SEEK_END)
    Code_Weigth = fo.tell()
    fo.close()
    size_compress.append(100 * Code_Weigth / Base_weight)

Conversion_rate = [float(x+1)*100/Height for x in range(Range-1)]

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.title('Vr_score')
plt.ylabel('Variance ratio')
plt.plot(Conversion_rate, vr_score, 'r-o')

plt.subplot(2, 2, 2)
plt.title("Psnr_score")
plt.ylabel('PSMR[db]')
plt.plot(Conversion_rate, psnr_score, 'r-o')

plt.subplot(2, 2, 3)
plt.title("Ssim_score")
plt.ylabel('SSIM')
plt.xlabel('Conversion_rate[%]')
plt.plot(Conversion_rate, ssim_score, 'r-o')

plt.subplot(2, 2, 4)
plt.title("Mse_score")
plt.ylabel('MSE')
plt.xlabel('Conversion_rate[%]')
plt.plot(Conversion_rate, mse_score, 'r-o')

fig_1 = plt.figure()
plt.subplot(2, 1, 1)
plt.title("Decompressed_Image")
plt.ylabel('Ratio[%]')
plt.plot(Conversion_rate, size, 'r-o')

plt.subplot(2, 1, 2)
plt.title("Compressed_Image")
plt.ylabel('Ratio[%]')
plt.xlabel('Conversion_rate[%]')
plt.plot(Conversion_rate, size_compress, 'r-o')

plt.show()
