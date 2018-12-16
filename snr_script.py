#test
# czyta zbiór testowy z folderu na podstawie folderów tworzy sobie klasy
# wycina cyfry z nazw folderów /klas/, opcjonalnie może też wyciąć reszte słów poza pierwszym /większa generalizacja, uogólnienie klas/
# wyznaczanie lbp z obrazów
# 2 listy
#   descryptory lbp
#   labele w postaci nazwy folderu
# Tworzenie zbioru uczącego
# dzielenie go losowo na zbiór uczący i zbiór walidujący
# kowersja listy labeli z word list na number list np. 'lemon' -> 20
# konwesja labeli z listy nuber na listy wektorów labeli 20 -> [0 0 0 0 0 ... 0 1 0 0 0 0 ...]
# merge listy list w jedną długą listę
# konwesja listy na array z numpy

# deskrypory zostały wyznaczone wczęniej i zapisane do pliku wystarczy tylko odczytać je


import os
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from skimage import feature
from sklearn.model_selection import train_test_split

TRAINING_PATH = "./fruits/fruits-360/Training/"
TEST_PATH = "./fruits/fruits-360/Test/"

# change
# for Windows            "\\"
# for Linux and MacOSX   "/"
NUM_OF_POINTS = 24
RADIUS = 8
NUM_OF_INPUTS = NUM_OF_POINTS + 2
MULTIPLY_NUM_OF_NEURONS = 2
NUM_OF_EPOCH = 100
BATH_SIZE = 100


def multi_layer_perceptron_gesheft(number_of_class, x_teach, y_teach, x_val, y_val, num_of_epoch, batch_size):
    model = Sequential()
    # first layer
    num_of_neurons = NUM_OF_INPUTS * MULTIPLY_NUM_OF_NEURONS
    model.add(Dense(num_of_neurons, input_shape=(NUM_OF_INPUTS,)))
    num_of_neurons *= MULTIPLY_NUM_OF_NEURONS

    # add how many layers you want
    # model.add(Dense(num_of_neurons,  init='uniform', activation='relu'))
    model.add(Dense(num_of_neurons))

    num_of_neurons *= MULTIPLY_NUM_OF_NEURONS
    model.add(Dense(num_of_neurons))
    num_of_neurons *= MULTIPLY_NUM_OF_NEURONS
    model.add(Dense(num_of_neurons))
    num_of_neurons *= MULTIPLY_NUM_OF_NEURONS
    model.add(Dense(num_of_neurons))

    # last layer
    model.add(Dense(number_of_class, activation='softmax'))

    # compile model: category classifier /many categories/, for Adam optimiser /Pacut approve/, category accuracy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    # Epochs (nb_epoch) is the number of times that the model is exposed to the training dataset.
    # Batch Size (batch_size) is the number of training instances shown to the model before a weight update is performed.
    # verbose show status bar
    results = model.fit(x_teach, y_teach, epochs=num_of_epoch, batch_size=batch_size, validation_data=(x_val, y_val),
                        verbose=1)

    show_summary_of_model(model, results)


def show_summary_of_model(model, results):
    print(model.summary())
    print(results.history.keys())
    # summarize history for accuracy
    plt.plot(results.history['categorical_accuracy'])
    plt.plot(results.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def read_samples(path):
    # list of pictures
    x = []
    # list of labels
    y = []

    # read path of directory
    dirpath = os.getcwd()
    # read subfolders
    subfolders = os.listdir(path)

    # delete numbers from string
    # result = ''.join(i for i in s if not i.isdigit())
    for folder in subfolders:
        # delete numbers from string
        # labels with variety types of fruit class /many types of apples/
        label = ''.join(i for i in folder if not i.isdigit())
        # get only first word /generalisation ex. only one label for apples - Apple /
        # label = label.partition(" ")[0]
        # print(label)
        tmp_path2_img = path + "/" + folder
        images = os.listdir(tmp_path2_img)
        count = 0
        for image in images:
            # print(image)
            lbp = compute_lbp(tmp_path2_img, image)
            # print(lbp)
            x.append(lbp)
            y.append(label)
            # read only x image from folder
            if count == 10:
                break
            count += 1
    return x, y


def local_binary_patterns(image, num_points, radius):
    # lbp = skimage.feature.local_binary_pattern(image, NUM_OF_POINTS, RADIUS, method="uniform")
    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    # normalize the histogram
    eps = 1e-7
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # return lbp
    return hist


def compute_lbp(image_path, image_name):
    image = cv2.imread(image_path + "/" + image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_patterns(gray, NUM_OF_POINTS, RADIUS)
    return lbp


def create_numpy_set(path, name):
    histogram, labels = read_samples(path)
    np.save("dataHistogram-" + name + ".npy", histogram)
    np.save("dataLabels-" + name + ".npy", labels)


def check_numpy_set_exist(name):
    if os.path.isfile("dataHistogram-" + name + ".npy") and os.path.isfile("dataLabels-" + name + ".npy"):
        return True
    return False


def load_set(name_desc, name_label):
    x = np.load(name_desc)
    y = np.load(name_label)
    return x, y


def load_numpy_set(name):
    x = np.load("dataX-" + name + ".npy")
    y = np.load("dataY-" + name + ".npy")
    return x, y


def make_id_class_dictionary(y_val):
    # make dictionary of classes
    y_val.sort()
    y_val = Counter(y_val).keys()

    class_id = {}
    temp_list_id = []
    count = 1

    for i in y_val:
        temp_list_id.append((i, count))
        count += 1

    for word, _id in temp_list_id:
        class_id[word] = _id
    print(class_id)

    return class_id


def cvt2_id_class_list(dictionary_class_id, word_list):
    # covert word list of class to id list of class
    y_teach = []
    for i in word_list:
        y_teach.append(dictionary_class_id[i])
    return y_teach


def cv2_id_class_as_vector(y_val_temp, num_of_class):
    y_val = []
    for i in y_val_temp:
        y = [0] * num_of_class
        y[i - 1] = 1
        y_val.append(y)
    np.asarray(y_val)
    return y_val


def main_processing_function():
    if not check_numpy_set_exist("teach"):
        create_numpy_set(TRAINING_PATH, "teach")
    x_teach, y_teach = load_numpy_set("teach")
    if not check_numpy_set_exist("val"):
        create_numpy_set(TEST_PATH, "val")
    x_val_temp, y_val_temp = load_numpy_set("val")

    # random Split Data set to teach
    x_teach, x_val, y_teach_temp1, y_val_temp1 = train_test_split(x_teach, y_teach, test_size=0.3, random_state=42)

    # number of class in set
    num_of_class = len(set(y_val_temp))

    # make idClasses
    id_class_dictionary = make_id_class_dictionary(y_val_temp)

    # convert from wordList to numberList Of class
    y_teach_temp2 = cvt2_id_class_list(id_class_dictionary, y_teach_temp1)
    y_val_temp2 = cvt2_id_class_list(id_class_dictionary, y_val_temp1)

    # get final representation of labels as vector ex. [0 0 0 0 ..... 0 1 0 0 0] - outputs of network
    y_val = cv2_id_class_as_vector(y_val_temp2, num_of_class)
    y_teach = cv2_id_class_as_vector(y_teach_temp2, num_of_class)

    multi_layer_perceptron_gesheft(number_of_class=num_of_class,
                                   x_teach=x_teach, y_teach=np.asarray(y_teach), x_val=x_val, y_val=np.asarray(y_val),
                                   num_of_epoch=NUM_OF_EPOCH,
                                   batch_size=BATH_SIZE)


def main():
    main_processing_function()


if __name__ == "__main__":
    main()
