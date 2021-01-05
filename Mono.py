from PIL import Image
from sklearn import decomposition
from sklearn.datasets import load_sample_images
from skimage import measure
from skimage import metrics
import numpy as np
from matplotlib import pyplot as plt
import os

Max_conversion_rate = 100  # Set number -> 0-100 [%] of the original picture size.
Picture_choice = 0  # Selection photo / Set 0/1


Picture = "Base_Mono.jpg"
Picture_output = "Mono_output.jpg"
Picture_compressed = "Mono_compressed_output.jpg"
mse_score, ssim_score, psnr_score, vr_score, size, size_compress = [], [], [], [], [], []

database = load_sample_images()
first_img = database.images[Picture_choice]
Show_img = Image.fromarray(first_img).convert('L')
Show_img.save(Picture)

Base_img = np.dot(first_img, [0.2989, 0.587, 0.114])
Base_img = np.float64(Base_img)
Height, Width = Base_img.shape[0], Base_img.shape[1]

f_original = open(Picture)
f_original.seek(0, os.SEEK_END)
Base_weight = f_original.tell()
f_original.close()

Range = int(Max_conversion_rate * Height / 100) if int(Max_conversion_rate * Height / 100) < Width else Width - 1 if Height == Width else Width
Range = 2 if Range == 0 else Range  # dla uzyskania min 1pkt

for n in range(1, Range):
    svd = decomposition.TruncatedSVD(n_components=n)
    Image_coded = svd.fit_transform(Base_img)
    Image_encoded = svd.inverse_transform(Image_coded)

    vr_score.append(svd.explained_variance_ratio_.sum())
    psnr_score.append(metrics.peak_signal_noise_ratio(Base_img, Image_encoded, data_range=255))
    mse_score.append(measure.simple_metrics.mean_squared_error(Base_img, Image_encoded))
    ssim_score.append(measure._structural_similarity.compare_ssim(Base_img, Image_encoded, data_range=255))

    # Zapisanany obraz z rozkompresowanej macierzy / porownanie rozmiaru do originalu
    new_im_1 = Image.fromarray(Image_encoded)
    new_im_1 = new_im_1.convert("L")
    new_im_1.save(Picture_output)
    fo = open(Picture_output)
    fo.seek(0, os.SEEK_END)
    size.append(100 * fo.tell() / Base_weight)
    fo.close()

    # Zapisana skompresowana macierz SVD / odtworzona w jpg i sprawdzanie jej rozmiaru
    new_im_2 = Image.fromarray(Image_coded)
    new_im_2 = new_im_2.convert("L")
    new_im_2.save(Picture_compressed)
    fo = open(Picture_compressed)
    fo.seek(0, os.SEEK_END)
    size_compress.append(100 * fo.tell() / Base_weight)
    fo.close()

Conversion_rate = [float(x+1)*100/Height for x in range(Range-1)]

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.title('Svd_score')
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
