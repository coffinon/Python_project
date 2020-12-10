import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as img
from sklearn.decomposition import TruncatedSVD
from PIL import Image

score, size, size_compress = [], [], []
Picture = "Maciej_Wielgosz.jpg"
Range = 511  # max range is width -1
str_output_image = "RGB_output.jpg"
str_output_com_imgage = "RGB_compressed_output.jpg"

image_array = np.array(Image.open(Picture))
image_original = open(Picture)
Original_size = image_original.seek(0, os.SEEK_END)
image_original.close()

print("Height:", image_array.shape[0], "Width:", image_array.shape[1])


for n in range(1, Range):

    svd = TruncatedSVD(n_components=n)
    RGB_full_array = []
    compressed_full_array = []
    for RGB_each_array in range(3):
        RGB = image_array[:, :, RGB_each_array]
        RGB_simple_array = svd.fit_transform(RGB)
        compressed_full_array.append(RGB_simple_array)

        # score.append(svd.explained_variance_ratio_.sum())
        RGB_full_array.append(svd.inverse_transform(RGB_simple_array))

    score.append(svd.explained_variance_ratio_.sum())

    array_out = np.dstack(RGB_full_array)
    array_out_compress = np.dstack(compressed_full_array)

    image_out = Image.fromarray(array_out.astype(np.uint8))
    new_im = image_out.convert("RGB")
    new_im.save(str_output_image)
    fo = open(str_output_image)
    fo.seek(0, os.SEEK_END)
    size.append(100 * fo.tell() / Original_size)
    fo.close()

    image_out = Image.fromarray(array_out_compress.astype(np.uint8))
    new_im = image_out.convert("RGB")
    new_im.save(str_output_com_imgage)
    fo = open(str_output_com_imgage)
    fo.seek(0, os.SEEK_END)
    size_compress.append(fo.tell())
    fo.close()

print("Original size:", Original_size, "bytes")

# plt.imshow(image_out/255)
# #check in matlib
# plt.show()

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.title('Accuracy')
plt.plot(range(1, len(score) + 1), score)

plt.subplot(2, 2, 2)
plt.title("Image size comparisons with original in %")
plt.plot(range(1, len(size) + 1), size)

plt.subplot(2, 1, 2)
plt.title("Saved decomposed data size")
plt.plot(range(1, len(size_compress) + 1), size_compress)
plt.show()
