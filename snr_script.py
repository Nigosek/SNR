import os
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from skimage import feature
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

TRAINING_PATH = "./fruits/fruits-360/Training/"
TEST_PATH = "./fruits/fruits-360/Test/"

# change
# for Windows            "\\"
# for Linux and MacOSX   "/"
NUM_OF_POINTS = 24
RADIUS = 8
NUM_OF_INPUTS = NUM_OF_POINTS + 2
MULTIPLY_NUM_OF_NEURONS = 4
NUM_OF_EPOCH = 50
BATH_SIZE = 1000

HIDDEN_LAYERS = 4
HIDDEN_LAYERS_WITHOUT_FIRST = HIDDEN_LAYERS - 1


def multi_layer_perceptron_gesheft(number_of_class, x_teach, y_teach, x_val, y_val, num_of_epoch, batch_size,
                                   hidden_layers_without_first):
    model = Sequential()
    # first layer
    num_of_neurons = NUM_OF_INPUTS * MULTIPLY_NUM_OF_NEURONS

    model.add(Dense(num_of_neurons, input_shape=(NUM_OF_INPUTS,)))

    # model.add(Dense(num_of_neurons, input_shape=(NUM_OF_INPUTS,)))

    for i in range(0, hidden_layers_without_first):
        model.add(Dense(num_of_neurons))

    # last layer
    model.add(Dense(number_of_class, activation='softmax'))

    # compile model: category classifier /many categories/, for Adam optimiser /Pacut approve/, category accuracy
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

    # Epochs (nb_epoch) is the number of times that the model is exposed to the training dataset.
    # Batch Size (batch_size) is the number of training instances shown to the model before a weight update is performed.
    # verbose show status bar
    results = model.fit(x_teach, y_teach, epochs=num_of_epoch, batch_size=batch_size, validation_data=(x_val, y_val),
                        verbose=1)

    return results, model


def show_summary_of_model(model, results):
    print(model.summary())
    print(results.history.keys())
    # summarize history for accuracy
    plt.plot(results.history['categorical_accuracy'])
    plt.plot(results.history['val_categorical_accuracy'])
    plt.title('model accuracy epoch=' + str(NUM_OF_EPOCH) + ' layers=' + str(HIDDEN_LAYERS))
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.gcf().clear()
    # summarize history for loss
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss epoch=' + str(NUM_OF_EPOCH) + ' layers=' + str(HIDDEN_LAYERS))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.gcf().clear()


def show_summary_of_model_all_test(results, data, append):
    directory = "./images/"

    legend = []
    for x in data:
        trainingData = data[x].get('categorical_accuracy')
        valiadationData = data[x].get('val_categorical_accuracy')
        tmp, = plt.plot(trainingData, linewidth=0.3, label=str(x + 1) + '-training')
        tmp2, = plt.plot(valiadationData, linewidth=0.3, label=str(x + 1) + '-test')
        legend.append(tmp)
        legend.append(tmp2)

        print(x)
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(handles=legend)
    plt.savefig(directory + append + '_model_accuracy_epoch_img_training222_test', format='eps', dpi=1000)
    plt.gcf().clear()

    legend = []
    for x in data:
        trainingData = data[x].get('loss')
        valiadationData = data[x].get('val_loss')
        tmp, = plt.plot(trainingData, linewidth=0.3, label=str(x + 1) + '-training')
        tmp2, = plt.plot(valiadationData, linewidth=0.3, label=str(x + 1) + '-test')
        legend.append(tmp)
        legend.append(tmp2)

        print(x)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(handles=legend)
    plt.savefig(directory + append + '_model_loss_epoch_img_training222_test', format='eps', dpi=1000)
    plt.gcf().clear()

    legend = []
    for x in data:
        trainingData = data[x].get('top_k_categorical_accuracy')
        valiadationData = data[x].get('val_top_k_categorical_accuracy')
        tmp, = plt.plot(trainingData, linewidth=0.3, label=str(x + 1) + '-training')
        tmp2, = plt.plot(valiadationData, linewidth=0.3, label=str(x + 1) + '-test')
        legend.append(tmp)
        legend.append(tmp2)

        print(x)
    plt.ylabel('top_5_categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(handles=legend)
    plt.savefig(directory + append + '_model_top5_accuracy_epoch_img_training222_test', format='eps', dpi=1000)
    plt.gcf().clear()


def show_summary_of_model_all(results, data, append):
    directory = "./images/"

    legend = []
    for x in data:
        trainingData = data[x].get('categorical_accuracy')
        tmp, = plt.plot(trainingData, linewidth=0.3, label=str(x + 1) + '-training')
        legend.append(tmp)

        print(x)
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(handles=legend)
    plt.savefig(directory + append + '_model_accuracy_epoch_img_training', format='eps', dpi=1000)
    plt.gcf().clear()

    legend = []
    for x in data:
        valiadationData = data[x].get('val_categorical_accuracy')
        tmp2, = plt.plot(valiadationData, linewidth=0.3, label=str(x + 1) + '-validation')
        legend.append(tmp2)

        print(x)
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(handles=legend)
    plt.savefig(directory + append + '_model_accuracy_epoch_img_validation', format='eps', dpi=1000)
    plt.gcf().clear()

    legend = []
    for x in data:
        # legend.append(x)
        trainingData = data[x].get('loss')
        tmp, = plt.plot(trainingData, linewidth=0.3, label=str(x + 1) + '-training')
        legend.append(tmp)

        print(x)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(handles=legend)
    plt.savefig(directory + append + '_model_loss_epoch_img_training', format='eps', dpi=1000)
    plt.gcf().clear()

    legend = []
    for x in data:
        trainingData = data[x].get('loss')
        valiadationData = data[x].get('val_loss')
        tmp2, = plt.plot(valiadationData, linewidth=0.3, label=str(x + 1) + '-validation')
        legend.append(tmp2)

        print(x)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(handles=legend)
    plt.savefig(directory + append + '_model_loss_epoch_img_validation', format='eps', dpi=1000)
    plt.gcf().clear()

    legend = []
    for x in data:
        trainingData = data[x].get('top_k_categorical_accuracy')
        valiadationData = data[x].get('val_top_k_categorical_accuracy')
        tmp, = plt.plot(trainingData, linewidth=0.3, label=str(x + 1) + '-training')
        legend.append(tmp)

        print(x)
    plt.ylabel('top_5_categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(handles=legend)
    plt.savefig(directory + append + '_model_top5_accuracy_epoch_img_training', format='eps', dpi=1000)
    plt.gcf().clear()

    legend = []
    for x in data:
        trainingData = data[x].get('top_k_categorical_accuracy')
        valiadationData = data[x].get('val_top_k_categorical_accuracy')
        tmp2, = plt.plot(valiadationData, linewidth=0.3, label=str(x + 1) + '-validation')
        legend.append(tmp2)

        print(x)
    plt.ylabel('top_5_categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(handles=legend)
    plt.savefig(directory + append + '_model_top5_accuracy_epoch_img_validation', format='eps', dpi=1000)
    plt.gcf().clear()

    plt.gcf().clear()


def save_summary_of_model(model, results, hiddend_layers):
    print("aaaaa")
    print(model.summary())
    print("bbbbb")
    print(results.history.keys())
    dir = "./images/"

    plt.plot(results.history['categorical_accuracy'])
    plt.plot(results.history['val_categorical_accuracy'])
    plt.title('model accuracy epoch=' + str(NUM_OF_EPOCH) + ' layers=' + str(hiddend_layers))
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # save image to file
    plt.savefig(
        dir + 'model_accuracy_epoch/' 'model_accuracy_epoch=' + str(NUM_OF_EPOCH) + ';layers=' + str(hiddend_layers))
    plt.gcf().clear()

    # summarize history for loss
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss epoch=' + str(NUM_OF_EPOCH) + ' layers=' + str(hiddend_layers))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    # save image to file
    plt.savefig(dir + 'model_loss_epoch/' + 'model_loss_epoch=' + str(NUM_OF_EPOCH) + ';layers=' + str(hiddend_layers))
    plt.gcf().clear()


def read_samples(path):
    # list of pictures
    x = []
    # list of labels
    y = []

    # read path of directory
    dirpath = os.getcwd()
    # read subfolders
    subfolders = os.listdir(path)

    print("*************************")
    print("read_samples:")
    print("NUM_OF_POINTS=" + str(NUM_OF_POINTS))
    print("RADIUS=" + str(RADIUS))

    # delete numbers from string
    # result = ''.join(i for i in s if not i.isdigit())
    for folder in subfolders:
        # delete numbers from string
        # labels with variety types of fruit class /many types of apples/
        label = ''.join(i for i in folder if not i.isdigit())
        # get only first word /generalisation ex. only one label for apples - Apple /
        tmp_path2_img = path + "/" + folder
        images = os.listdir(tmp_path2_img)
        for image in images:
            lbp = compute_lbp(tmp_path2_img, image)
            x.append(lbp)
            y.append(label)
    return x, y


def local_binary_patterns(image, num_points, radius):
    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    # normalize the histogram
    eps = 1e-7
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def compute_lbp(image_path, image_name):
    image = cv2.imread(image_path + "/" + image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_patterns(gray, NUM_OF_POINTS, RADIUS)
    return lbp


def create_numpy_set(path, name):
    histogram, labels = read_samples(path)
    np.save("dataX-" + name + ".npy", histogram)
    np.save("dataY-" + name + ".npy", labels)


def check_numpy_set_exist(name):
    if os.path.isfile("dataX-" + name + ".npy") and os.path.isfile("dataY-" + name + ".npy"):
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
    if not check_numpy_set_exist("test"):
        create_numpy_set(TEST_PATH, "test")
    x_test_temp, y_test_temp = load_numpy_set("test")
    y_test_temp_copy = y_test_temp.copy()

    # random Split Data set to teach
    x_teach, x_val, y_teach_temp1, y_val_temp1 = train_test_split(x_teach, y_teach, test_size=0.3, random_state=42)

    # number of class in set
    num_of_class = len(set(y_test_temp_copy))

    # make idClasses
    id_class_dictionary = make_id_class_dictionary(y_test_temp_copy)

    # convert from wordList to numberList Of class
    y_teach_temp2 = cvt2_id_class_list(id_class_dictionary, y_teach_temp1)
    y_val_temp2 = cvt2_id_class_list(id_class_dictionary, y_val_temp1)

    # get final representation of labels as vector ex. [0 0 0 0 ..... 0 1 0 0 0] - outputs of network
    y_val = cv2_id_class_as_vector(y_val_temp2, num_of_class)
    y_teach = cv2_id_class_as_vector(y_teach_temp2, num_of_class)

    results, model = multi_layer_perceptron_gesheft(number_of_class=num_of_class,
                                                    x_teach=x_teach, y_teach=np.asarray(y_teach), x_val=x_val,
                                                    y_val=np.asarray(y_val),
                                                    num_of_epoch=NUM_OF_EPOCH,
                                                    batch_size=BATH_SIZE,
                                                    hidden_layers_without_first=HIDDEN_LAYERS_WITHOUT_FIRST)
    show_summary_of_model(model, results)

    # evaluate_test
    y_test_tmp = cvt2_id_class_list(id_class_dictionary, y_test_temp)
    y_test = cv2_id_class_as_vector(y_test_tmp, num_of_class)
    evaluate_test = model.evaluate(x_test_temp, np.asarray(y_test), verbose=0)
    print("********************")
    # Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics).
    # The attribute model.metrics_names will give you the display labels for the scalar outputs.
    print(model.metrics_names)
    print(evaluate_test)

    # ROC
    col = 'micro'
    mi_avg_fprs = []
    mi_avg_tprs = []

    y_test_array = np.asarray(y_test)

    models = [model]
    roc_plot_data(models, x_test_temp, y_test_array, num_of_class, mi_avg_fprs, mi_avg_tprs)
    mi_avg_fprs = np.array(mi_avg_fprs)
    mi_avg_tprs = np.array(mi_avg_tprs)

    for idx in range(0, 4):
        plt.subplot(4 / 2, 4 / 2, idx + 1)
        roc_plot(0, mi_avg_fprs, mi_avg_tprs)
    plt.tight_layout(h_pad=1.0, w_pad=1.0)
    plt.show()

    # *******************************************************************************
    # test_data
    y_test_tmp = cvt2_id_class_list(id_class_dictionary, y_test_temp)
    y_test = cv2_id_class_as_vector(y_test_tmp, num_of_class)

    results, model = multi_layer_perceptron_gesheft(number_of_class=num_of_class,
                                                    x_teach=x_teach, y_teach=np.asarray(y_teach), x_val=x_test_temp,
                                                    y_val=np.asarray(y_test),
                                                    num_of_epoch=NUM_OF_EPOCH,
                                                    batch_size=BATH_SIZE,
                                                    hidden_layers_without_first=HIDDEN_LAYERS_WITHOUT_FIRST)
    show_summary_of_model(model, results)

    # evaluate_test
    y_test_tmp = cvt2_id_class_list(id_class_dictionary, y_test_temp)
    y_test = cv2_id_class_as_vector(y_test_tmp, num_of_class)
    evaluate_test = model.evaluate(x_test_temp, np.asarray(y_test), verbose=0)
    print("********************")
    # Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics).
    # The attribute model.metrics_names will give you the display labels for the scalar outputs.
    print(model.metrics_names)
    print(evaluate_test)

    # ROC
    col = 'micro'
    mi_avg_fprs = []
    mi_avg_tprs = []
    # mi_avg_aucs = []

    y_test_array = np.asarray(y_test)

    models = [model]
    roc_plot_data(models, x_test_temp, y_test_array, num_of_class, mi_avg_fprs, mi_avg_tprs)
    mi_avg_fprs = np.array(mi_avg_fprs)
    mi_avg_tprs = np.array(mi_avg_tprs)

    for idx in range(0, 4):
        plt.subplot(4 / 2, 4 / 2, idx + 1)
        roc_plot(0, mi_avg_fprs, mi_avg_tprs)
    plt.tight_layout(h_pad=1.0, w_pad=1.0)
    plt.show()


def roc_plot_data(models, x_test_temp, y_test_array, num_of_class, mi_avg_fprs, mi_avg_tprs):
    for model in models:
        col = 'micro'
        y_pred = model.predict(x_test_temp)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_of_class):
            fpr[i], tpr[i], _ = roc_curve(y_test_array[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_array.ravel(), y_pred.ravel())
        # Micro-avarage Area Under Curve
        roc_auc[col] = auc(fpr[col], tpr[col])

        mi_avg_fprs.append(fpr[col])
        mi_avg_tprs.append(tpr[col])


def roc_plot(idx, mi_avg_fprs, mi_avg_tprs):
    plt.plot(mi_avg_fprs[idx], mi_avg_tprs[idx], color='darkorange',
             lw=2)
    plt.title(str(idx))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPr')
    plt.ylabel('TPr')


def multiplyLayersTest(number_of_layers=3):
    if not check_numpy_set_exist("teach"):
        create_numpy_set(TRAINING_PATH, "teach")
    x_teach, y_teach = load_numpy_set("teach")
    if not check_numpy_set_exist("test"):
        create_numpy_set(TEST_PATH, "test")
    x_test_temp, y_test_temp = load_numpy_set("test")
    y_test_temp_copy = y_test_temp.copy()

    # random Split Data set to teach
    x_teach, x_val, y_teach_temp1, y_val_temp1 = train_test_split(x_teach, y_teach, test_size=0.3, random_state=42)

    # number of class in set
    num_of_class = len(set(y_test_temp_copy))

    # make idClasses
    id_class_dictionary = make_id_class_dictionary(y_test_temp_copy)

    # convert from wordList to numberList Of class
    y_teach_temp2 = cvt2_id_class_list(id_class_dictionary, y_teach_temp1)
    y_val_temp2 = cvt2_id_class_list(id_class_dictionary, y_val_temp1)

    # get final representation of labels as vector ex. [0 0 0 0 ..... 0 1 0 0 0] - outputs of network
    y_val = cv2_id_class_as_vector(y_val_temp2, num_of_class)
    y_teach = cv2_id_class_as_vector(y_teach_temp2, num_of_class)

    data_for_plots = {}
    models = []
    number_of_iterations = number_of_layers
    plt.figure(figsize=(10, 10))
    for i in range(0, number_of_iterations):
        results, model = multi_layer_perceptron_gesheft(number_of_class=num_of_class,
                                                        x_teach=x_teach, y_teach=np.asarray(y_teach), x_val=x_val,
                                                        y_val=np.asarray(y_val),
                                                        num_of_epoch=NUM_OF_EPOCH,
                                                        batch_size=BATH_SIZE,
                                                        hidden_layers_without_first=i)
        data_for_plots[i] = results.history

        # evaluate_test
        y_test_tmp = cvt2_id_class_list(id_class_dictionary, y_test_temp)
        y_test = cv2_id_class_as_vector(y_test_tmp, num_of_class)
        evaluate_test = model.evaluate(x_test_temp, np.asarray(y_test), verbose=0)
        print("********************")
        # Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics).
        # The attribute model.metrics_names will give you the display labels for the scalar outputs.
        print(model.metrics_names)
        print(evaluate_test)

        # ROC
        col = 'micro'
        mi_avg_fprs = []
        mi_avg_tprs = []

        y_test_array = np.asarray(y_test)

        models.append(model)
        roc_plot_data(models, x_test_temp, y_test_array, num_of_class, mi_avg_fprs, mi_avg_tprs)
        mi_avg_fprs = np.array(mi_avg_fprs)
        mi_avg_tprs = np.array(mi_avg_tprs)

    # wykresy dla ROC, dla 6 wyk -> 2x3, dla 9 -> 3x3, więcej rand
    for idx in range(0, number_of_iterations):
        # if(number_of_iterations%2)
        if number_of_iterations % 2 == 0:
            tmp = number_of_iterations / 2
        else:
            tmp = number_of_iterations / 2 + 1
        if number_of_iterations == 1:
            plt.subplot(1, 1, idx + 1)
        elif number_of_iterations <= 6:
            plt.subplot(2, tmp, idx + 1)
        elif number_of_iterations <= 9:
            plt.subplot(3, 3, idx + 1)
        else:
            plt.subplot(3, tmp, idx + 1)

        roc_plot(idx, mi_avg_fprs, mi_avg_tprs)
    plt.tight_layout(h_pad=1.0, w_pad=1.0)
    plt.show()
    plt.gcf().clear()

    show_summary_of_model_all(results, data_for_plots, 'NUM_OF_LAYERS')

    # ***************************************
    # test set
    y_test_tmp = cvt2_id_class_list(id_class_dictionary, y_test_temp)
    y_test = cv2_id_class_as_vector(y_test_tmp, num_of_class)
    data_for_plots = {}
    for i in range(0, number_of_iterations):
        results, model = multi_layer_perceptron_gesheft(number_of_class=num_of_class,
                                                        x_teach=x_teach, y_teach=np.asarray(y_teach), x_val=x_test_temp,
                                                        y_val=np.asarray(y_test),
                                                        num_of_epoch=NUM_OF_EPOCH,
                                                        batch_size=BATH_SIZE,
                                                        hidden_layers_without_first=i)

        data_for_plots[i] = results.history

    show_summary_of_model_all_test(results, data_for_plots, 'NUM_OF_LAYERS')


def removeFile(name):
    if os.path.exists(name):
        os.remove(name)
    else:
        print("The file does not exist")


def LBPTest():
    # changing NUM_OF_POINTS and RADIUS value
    global NUM_OF_POINTS
    global RADIUS
    global NUM_OF_INPUTS
    data_for_plots = {}

    for i in range(4, 36, 2):
        NUM_OF_POINTS = i
        RADIUS = 8
        NUM_OF_INPUTS = NUM_OF_POINTS + 2
        print("****************************")
        print("LBPTest")
        print("NUM_OF_POINTS=" + str(NUM_OF_POINTS))
        print("RADIUS=" + str(RADIUS))

        removeFile("dataX-teach.npy")
        removeFile("dataX-test.npy")
        removeFile("dataY-teach.npy")
        removeFile("dataY-test.npy")
        # if not check_numpy_set_exist("teach"):
        create_numpy_set(TRAINING_PATH, "teach")
        x_teach, y_teach = load_numpy_set("teach")
        # if not check_numpy_set_exist("test"):
        create_numpy_set(TEST_PATH, "test")
        x_test_temp, y_test_temp = load_numpy_set("test")
        y_test_temp_copy = y_test_temp.copy()

        # random Split Data set to teach
        x_teach, x_val, y_teach_temp1, y_val_temp1 = train_test_split(x_teach, y_teach, test_size=0.3, random_state=42)

        # number of class in set
        num_of_class = len(set(y_test_temp_copy))

        # make idClasses
        id_class_dictionary = make_id_class_dictionary(y_test_temp_copy)

        # convert from wordList to numberList Of class
        y_teach_temp2 = cvt2_id_class_list(id_class_dictionary, y_teach_temp1)
        y_val_temp2 = cvt2_id_class_list(id_class_dictionary, y_val_temp1)

        # get final representation of labels as vector ex. [0 0 0 0 ..... 0 1 0 0 0] - outputs of network
        y_val = cv2_id_class_as_vector(y_val_temp2, num_of_class)
        y_teach = cv2_id_class_as_vector(y_teach_temp2, num_of_class)

        # for i in range(0, 8):
        results, model = multi_layer_perceptron_gesheft(number_of_class=num_of_class,
                                                        x_teach=x_teach, y_teach=np.asarray(y_teach), x_val=x_val,
                                                        y_val=np.asarray(y_val),
                                                        num_of_epoch=NUM_OF_EPOCH,
                                                        batch_size=BATH_SIZE,
                                                        hidden_layers_without_first=HIDDEN_LAYERS_WITHOUT_FIRST)
        data_for_plots[i - 1] = results.history
    show_summary_of_model_all(results, data_for_plots, 'NUM_OF_POINTS')


def main():
    # main_processing_function()
    # multiplyLayersTest(10)
    LBPTest()


if __name__ == "__main__":
    main()
