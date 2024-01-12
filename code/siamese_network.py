import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.models import Model


def initialize_bias(shape, name=None, dtype=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def last_layer(encoded_l, encoded_r, lyr_name='cos'):
    if lyr_name == 'L1':
        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors: tf.keras.backend.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        add_dens = Dense(512, activation='relu', bias_initializer=initialize_bias)(L1_distance)
        # drp_lyr = Dropout(0.25)(add_dens)
        # xx = Dense(128, activation='relu', bias_initializer=initialize_bias)(add_dens)
        # prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
        prediction = Dense(1, activation='sigmoid')(xx)

    elif lyr_name == 'L2':
        # Write L2 here
        L2_layer = Lambda(lambda tensors: (tensors[0] - tensors[1]) ** 2 / (tensors[0] + tensors[1]))
        L2_distance = L2_layer([encoded_l, encoded_r])
        add_dens = Dense(512, activation='relu', bias_initializer=initialize_bias)(L2_distance)
        drp_lyr = Dropout(0.25)(add_dens)
        # xx =  Dense(128, activation='relu', bias_initializer=initialize_bias)(drp_lyr)
        # drp_lyr2 = Dropout(0.25)(xx)
        # x =  Dense(64, activation='relu', bias_initializer=initialize_bias)(xx)
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(drp_lyr)

    else:
        # Add cosine similarity function
        cos_layer = Lambda(lambda tensors: K.sum(tensors[0] * tensors[1], axis=-1, keepdims=True) /
                                           tf.keras.backend.l2_normalize(tensors[0]) * tf.keras.backend.l2_normalize(
            tensors[1]))
        cos_distance = cos_layer([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(cos_distance)

    return prediction


def create_couples(x_support, y_support, x_train, y_train):
    x_train_left = []
    x_train_right = []
    y_train_set = []
    for i in range(0, len(x_support)):
        for j in range(0, len(x_train)):
            x_train_left.append(x_support[i])
            x_train_right.append(x_train[j])
            if y_support[i] == y_train[j]:
                y_train_set.append(1)
            else:
                y_train_set.append(0)

    return np.array(x_train_left), np.array(x_train_right), np.array(y_train_set)


def eval_siamese_model(loss, accuracy, precision, recall, auc, classes):
    results = pd.DataFrame(columns=["Cancer", "Loss", "Accuracy", "Precision", "Recall", "AUC", "F1 Score"])
    for cancer in classes:
        f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
        new_row = {"Cancer": cancer, "Loss": loss, "Accuracy": accuracy, "Precision": precision,
                   "Recall": recall, "AUC": auc, "F1 Score": f1_score}
        results = pd.concat([results, pd.DataFrame.from_dict([new_row])])

    return results


def created_model_siamese(input_shape):
    input_layer = Input(shape=input_shape)
    conv1 = Conv1D(filters=256, kernel_size=50, strides=50, activation='relu', padding='same')(input_layer)
    conv2 = Conv1D(filters=128, kernel_size=10, strides=1, activation='relu', padding='same')(conv1)
    maxpool1 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='same')(maxpool1)
    maxpool2 = MaxPooling1D(pool_size=2)(conv3)

    output = Flatten()(maxpool2)

    return output, input_layer


def make_oneshot_task(genes_len, x_val, x_test, classes, class_test_ind, class_val_ind, N, s="test"):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'val':
        X = x_val.values
        class_test_dic = class_val_ind
    else:
        X = x_test.values
        class_test_dic = class_test_ind

    N = len(class_test_dic.keys())
    list_N_samples = random.sample(list(class_test_dic.keys()), N)
    true_category = list_N_samples[0]
    out_ind = np.array([random.sample(class_test_dic[j], 2) for j in list_N_samples])
    indices = out_ind[:, 1]
    ex1 = out_ind[0, 0]

    # create one column of one sample
    test_image = np.asarray([X[ex1]] * N).reshape(N, genes_len, 1)
    support_set = X[indices].reshape(N, genes_len, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set, list_N_samples = shuffle(targets, test_image, support_set, list_N_samples)
    pairs = [test_image, support_set]

    return pairs, targets, true_category, list_N_samples


def test_oneshot(model, genes_len, x_val, x_test, classes, class_test_ind, class_val_ind, N, k, s="test", verbose=0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
    for i in range(k):
        inputs, targets, true_category, list_N_samples = make_oneshot_task(genes_len, x_val, x_test, classes,
                                                                           class_test_ind, class_val_ind, N, s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
    return percent_correct


def get_batch(batch_size, x_train, class_train_ind, genes_len, md='train'):
    """
    Create batch of n pairs, half same class, half different class
    """
    categories = random.sample(list(class_train_ind.keys()), len(class_train_ind.keys()))
    n_classes = len(class_train_ind.keys())

    pairs = [np.zeros((batch_size, genes_len, 1)) for i in range(2)]
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    j = 0
    for i in range(batch_size):
        if j > n_classes - 1:
            categories = random.sample(list(class_train_ind.keys()), len(class_train_ind.keys()))
            j = 0

        category = categories[j]
        idx_1 = random.sample(class_train_ind[category], 1)[0]

        pairs[0][i, :, :] = x_train.values[idx_1].reshape(genes_len, 1)
        if i >= batch_size // 2:
            category_2 = category
            idx_2 = random.sample(class_train_ind[category_2], 1)[0]
            pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1)

        else:
            ind_pop = list(categories).index(category)
            copy_list = categories.copy()
            copy_list.pop(ind_pop)
            category_2 = random.sample(copy_list, 1)[0]
            idx_2 = random.sample(class_train_ind[category_2], 1)[0]
            pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1)

        j += 1
    return pairs, targets


def indices_save(dataset):
    # Creazione di una mappa dove la chiave è il tipo di cancro e
    # il valore è una lista degli indici delle righe
    cancer_map = {}
    for index, row in dataset.iterrows():
        cancer_type = row['CANCER_TYPE']
        if cancer_type in cancer_map:
            cancer_map[cancer_type].append(index)
        else:
            cancer_map[cancer_type] = [index]
    return cancer_map


def siamese_network(dataset_genes, model, input_shape, genes_len, classes, n_classes, cancer_type):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Modello per l'input sinistro
    print("Model left...")
    model_left_output, model_left_input = created_model_siamese(input_shape)
    model_left = Model(inputs=model_left_input, outputs=model_left_output)
    model_left.layers[1].set_weights(model.layers[1].get_weights())
    model_left.layers[2].set_weights(model.layers[2].get_weights())
    model_left.layers[4].set_weights(model.layers[4].get_weights())

    # Modello per l'input destro
    print("Model right...")
    model_right_output, model_right_input = created_model_siamese(input_shape)
    model_right = Model(inputs=model_right_input, outputs=model_right_output)
    model_right.layers[1].set_weights(model.layers[1].get_weights())
    model_right.layers[2].set_weights(model.layers[2].get_weights())
    model_right.layers[4].set_weights(model.layers[4].get_weights())

    # Collegamento dei modelli
    print("Merge...")
    encoded_l = model_left(left_input)
    encoded_r = model_right(right_input)

    # Creazione del modello Siamese
    print("Create siamese model...")
    siamese_output = last_layer(encoded_l, encoded_r, lyr_name='L2')
    siamese_model = Model(inputs=[left_input, right_input], outputs=siamese_output)

    print("Compile in progress...")
    optimizer = Adam(lr=0.000005)
    siamese_model.compile(loss="binary_crossentropy", optimizer=optimizer)

    # Hyper parameters
    evaluate_every = 200  # interval for evaluating on one-shot tasks
    batch_size = 128  # max 12 for 19
    n_iter = 20000  # No. of training iterations
    N_way = 22  # how many classes for testing one-shot tasks
    n_val = 1000  # how many one-shot tasks to validate on
    best = -1

    # Pre-Processing dataset for siamese network
    dataset_genes["CANCER_TYPE"] = cancer_type

    # Ripetiamo train_test_split finché x_test non contiene almeno due valori 'Nerve Sheath Tumor' in 'CANCER_TYPE'
    while True:
        random_state = random.randint(0, 42)
        x_train, x_test = train_test_split(dataset_genes,
                                           test_size=0.20, stratify=cancer_type, random_state=random_state)

        # Controllare se ci sono almeno due 'Nerve Sheath Tumor' in x_test
        if (x_test['CANCER_TYPE'] == 'Nerve Sheath Tumor').sum() >= 2:
            break

    # x_train, x_val = train_test_split(x_train, test_size=0.15, random_state=42)
    x_val = pd.DataFrame()

    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    # x_val = x_val.reset_index(drop=True)

    class_train_ind = indices_save(x_train)
    # class_val_ind = indices_save(x_val)
    class_val_ind = None
    class_test_ind = indices_save(x_test)

    x_train = x_train.drop('CANCER_TYPE', axis=1)
    x_test = x_test.drop('CANCER_TYPE', axis=1)
    # x_val = x_val.drop('CANCER_TYPE', axis=1)

    print(len(class_train_ind.keys()))
    print("Starting training process!")
    for i in range(1, n_iter + 1):
        (inputs, targets) = get_batch(batch_size, x_train, class_train_ind, genes_len)
        loss = siamese_model.train_on_batch(inputs, targets)

        if i % 100 == 0:
            print(i)
        if i % evaluate_every == 0:
            print("\n ------------- \n")
            print("Train Loss: {0}".format(loss))
            val_acc = test_oneshot(siamese_model, genes_len, x_val, x_test, classes, class_test_ind, class_val_ind,
                                   N_way, n_val, s='test', verbose=True)
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                print(str(i))
                best = val_acc
