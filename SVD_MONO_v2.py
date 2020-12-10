from PIL import Image
from sklearn import decomposition
import numpy as np
from matplotlib import pyplot as plt
import os

score, size, size_compress = [], [], []

Picture = "Mono-test.jpg"
im_array = np.array(Image.open(Picture).convert('L'))
f_original = open(Picture)
f_original.seek(0, os.SEEK_END)
str_output_image = "Mono_output.jpg"
str_output_com_imgage = "Mono_compressed_output.jpg"
Range = 979  # MAX width -1 -> Stopien kompresji
#  if transpose MAX height -1

print("Height:", im_array.shape[0], "Width:", im_array.shape[1])

# aby tansponowac macierz odkomentuj w dwóch miejscach
# im_array=im_array.transpose()

for n in range(1, Range):
    svd = decomposition.TruncatedSVD(n_components=n)
    X_reduced = svd.fit_transform(im_array)
    score.append(svd.explained_variance_ratio_.sum())
    image_reduced = svd.inverse_transform(X_reduced)

    # image_reduced = image_reduced.transpose()

    # Zapisanany obraz z rozkompresowanej macierzy / porownanie rozmiaru do originalu
    new_im = Image.fromarray(image_reduced)
    new_im = new_im.convert("L")
    new_im.save(str_output_image)
    fo = open(str_output_image)
    fo.seek(0, os.SEEK_END)
    size.append(100 * fo.tell() / f_original.tell())
    fo.close()

    # Zapisana skompresowana macierz SVD / odtworzona w jpg i sprawdzanie jej rozmiaru
    new_im = Image.fromarray(X_reduced)
    new_im = new_im.convert("L")
    new_im.save(str_output_com_imgage)
    fo = open(str_output_com_imgage)
    fo.seek(0, os.SEEK_END)
    size_compress.append(fo.tell())
    fo.close()

print("Original size:", f_original.tell(), "bytes")

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.title('Accuracy')
plt.stem(range(1, Range), score)

plt.subplot(2, 2, 2)
plt.title("Image size comparisons with original in %")
plt.plot(range(1, Range), size)

plt.subplot(2, 1, 2)
plt.title("Saved decomposed data size")
plt.plot(range(1, Range), size_compress)
plt.show()
