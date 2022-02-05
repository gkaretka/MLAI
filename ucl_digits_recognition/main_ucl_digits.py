from os.path import exists
import keras
import matplotlib.pyplot
import numpy as np
from PIL import Image
from keras import layers
import pandas as pd
from sklearn.preprocessing import scale

model_name = "digits_model_4"
model_extension = ".h5"
img_path = "img/"
image_names = ["8_0.png", "7_0.jpg", "5_0.jpg", "2_0.jpg", "1_0.jpg"]

file_exists = exists(model_name + model_extension)
if file_exists:
    model: keras.Sequential
    model = keras.models.load_model(model_name + model_extension)
    new_pixel_values = []
    for i in range(len(image_names)):
        im = Image.open(img_path + image_names[i])
        pix_val = list(im.getdata())
        _new_pix = []
        for p_values in pix_val:
            avg = (p_values[0] + p_values[1] + p_values[2]) / 3.0
            avg = float(((avg * 16.0) / 255.0))
            _new_pix.append(avg)

        _new_pix = scale(_new_pix)*(-1)
        new_pixel_values.append(_new_pix)
        im.close()

    samples_to_predict = np.array(new_pixel_values)

    cnt = 0
    array_images = []
    for sample in samples_to_predict:
        array_img = []
        row = []
        for i in range(len(sample)):
            row.append(int(sample[i]))
            cnt = cnt + 1
            if cnt % 8 == 0 and cnt != 0:
                array_img.append(row)
                row = []
        array_images.append(array_img)

    results = model.predict(samples_to_predict)
    for i in range(len(results)):
        matplotlib.pyplot.imshow(array_images[i])
        matplotlib.pyplot.show()
        print(np.argmax(results[i]))

else:
    data_train = pd.read_csv("optdigits.tra")
    data_test = pd.read_csv("optdigits.tes")

    num_of_pixels = 8 * 8
    cls_index = 64

    # separate pixel values and class
    x_train = data_train.iloc[:, 0:num_of_pixels].values
    y_train = data_train.iloc[:, cls_index].values

    x_test = data_test.iloc[:, 0:num_of_pixels].values
    y_test = data_test.iloc[:, cls_index].values

    x_train = scale(x_train)
    x_test = scale(x_test)

    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(64, 1)))
    model.add(keras.layers.Dense(640, activation="sigmoid"))
    model.add(keras.layers.Dense(320, activation="linear"))
    model.add(keras.layers.Dense(640, activation="relu"))
    model.add(keras.layers.Dense(320, activation="linear"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    # sparse_categorical_crossentropy for integer output
    # categorical_crossentropy for one hot encoding
    model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

    model.fit(x_train, y_train, epochs=20, batch_size=10)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    model.save(model_name + model_extension)

"""
    model: keras.Sequential
    model = keras.models.load_model(model_name + model_extension)
    new_pixel_values = []
    for i in range(len(image_names)):
        im = Image.open(img_path + image_names[i])
        pix_val = list(im.getdata())
        _new_pix = []
        for p_values in pix_val:
            avg = (p_values[0] + p_values[1] + p_values[2])/3
            avg = int(((avg * 16.0) / 255.0))
            _new_pix.append(avg)

        _new_pix = scale(_new_pix)
        new_pixel_values.append(_new_pix)
        im.close()

    samples_to_predict = np.array(new_pixel_values)

    cnt = 0
    array_images = []
    for sample in samples_to_predict:
        array_img = []
        row = []
        for i in range(len(sample)):
            row.append(int(sample[i]))
            cnt = cnt + 1
            if cnt % 8 == 0 and cnt != 0:
                array_img.append(row)
                row = []
        array_images.append(array_img)

    print(array_images[0])
    matplotlib.pyplot.imshow(array_images[0])
    matplotlib.pyplot.show()

    results = model.predict(samples_to_predict)
    for result in results:
        print(np.argmax(result))
"""