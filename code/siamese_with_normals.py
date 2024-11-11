import random
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from jupyter_core.migrate import regex
from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.keras.utils.generic_utils import custom_object_scope
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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
        prediction = Dense(1, activation='sigmoid')()

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


def make_oneshot_task(genes_len, x_test, class_test_ind, N):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    X = x_test.values
    class_test_dic = class_test_ind

    list_N_samples = random.sample(list(class_test_dic.keys()), N)
    true_category = list_N_samples[0]
    out_ind = np.array([random.sample(class_test_dic[j], 2) for j in list_N_samples])
    indices = out_ind[:, 1]
    ex1 = out_ind[0, 0]

    # create one column of one sample
    test_image = np.asarray([X[ex1]] * N).reshape(N, genes_len, 1).astype(np.float128)
    support_set = X[indices].reshape(N, genes_len, 1).astype(np.float128)
    targets = np.zeros((N,), dtype= np.float128)
    targets[0] = 1
    targets, test_image, support_set, list_N_samples = shuffle(targets, test_image, support_set, list_N_samples)
    pairs = [test_image, support_set]

    return pairs, targets, true_category, list_N_samples


def test_oneshot(model, genes_len, x_test, class_test_ind, N, k, verbose=0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
    for i in range(k):
        inputs, targets, true_category, list_N_samples = make_oneshot_task(genes_len, x_test, class_test_ind, N)
        probs = model.predict(inputs, verbose=0)
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
    # Emanuele ver. : modifiche per togliere ,
    # x_train = x_train.replace(',', '.', regex=True)
    #
    # x_train = x_train.astype(np.float64)
    #
    categories = random.sample(list(class_train_ind.keys()), len(class_train_ind.keys()))
    n_classes = len(class_train_ind.keys())

    pairs = [np.zeros((batch_size, genes_len, 1)) for i in range(2)]

    targets = np.zeros((batch_size,), dtype= np.float128)

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    j = 0
    for i in range(batch_size):
        if j > n_classes - 1:
            categories = random.sample(list(class_train_ind.keys()), len(class_train_ind.keys()))
            j = 0

        category = categories[j]
        idx_1 = random.sample(class_train_ind[category], 1)[0]

        pairs[0][i, :, :] = x_train.values[idx_1].reshape(genes_len, 1).astype(np.float128)
        if i >= batch_size // 2:
            category_2 = category
            idx_2 = random.sample(class_train_ind[category_2], 1)[0]
            pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1).astype(np.float128)

        else:
            ind_pop = list(categories).index(category)
            copy_list = categories.copy()
            copy_list.pop(ind_pop)
            category_2 = random.sample(copy_list, 1)[0]
            idx_2 = random.sample(class_train_ind[category_2], 1)[0]
            pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1).astype(np.float128)

        j += 1
    return pairs, targets


def indices_save(dataset):
    """Creazione di una mappa dove la chiave è il tipo di cancro e
    il valore è una lista degli indici delle righe """
    cancer_map = {}
    for index, cancer_type in dataset['CANCER_TYPE'].items():
        if cancer_type in cancer_map:
            cancer_map[cancer_type].append(index)
        else:
            cancer_map[cancer_type] = [index]
    return cancer_map


def get_siamese_model(input_shape, model):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    siamese_model = Sequential()

    siamese_model.add(
        Conv1D(filters=256, kernel_size=50, strides=1, activation='relu', weights=model.layers[1].get_weights(),
               padding='same', input_shape=input_shape))  # lo strides di questo era impostato a 'strides=50'
    siamese_model.add(
        Conv1D(filters=128, kernel_size=10, strides=1, activation='relu', weights=model.layers[2].get_weights(),
               padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))

    siamese_model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='sigmoid',
                             weights=model.layers[4].get_weights(), padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))

    #######################################################################################
    # Ulterior 2 layers
    siamese_model.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='sigmoid',
                             weights=model.layers[6].get_weights(), padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))

    siamese_model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='sigmoid',
                             weights=model.layers[8].get_weights(), padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))
    #######################################################################################

    siamese_model.add(Flatten())

    encoded_l = siamese_model(left_input)
    encoded_r = siamese_model(right_input)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input],
                        outputs=last_layer(encoded_l, encoded_r, lyr_name='L2'),  # prediction and cosine_similarity
                        name="siamese-network")
    return siamese_net


# Emanuele: creata funzione separata per calcolare i treshold
def calc_threshold(siamese_model, dataset_genes, cancer_type, selected_type, start_index):
    # Per ogni tipo univoco di cancro, utilizzato come chiave del dizionario, vengono impostati
    # come valori di default [1(minimo), 0(massimo), [](array su cui calcolare medio) ,[] (array per la deviazione)]
    threshold = {selected_type: [1, 0, [], []]}

    # Scorro il dataset con i pazienti affetti a partire dall'indice del tipo di selected_type
    for k in range(start_index, len(dataset_genes.index)):
        cancer_key_k = cancer_type.values[k][0]
        for j in range(start_index, len(dataset_genes.index)):
            cancer_key_j = cancer_type.values[j][0]
            # Se il tipo di cancro è uguale inizio a calcolare il threshold
            if cancer_key_k == cancer_key_j and cancer_key_j == selected_type:
                input_uno = tf.convert_to_tensor(dataset_genes.iloc[k])
                input_uno = tf.expand_dims(input_uno, axis=0)

                input_due = tf.convert_to_tensor(dataset_genes.iloc[j])
                input_due = tf.expand_dims(input_due, axis=0)

                # Risultato del threshold temporaneo
                temp = siamese_model.predict([input_uno, input_due], verbose=False)[0][0]

                # Metto nella posizione 0 il minimo dei threshold e nella posizione 1 il massimo
                threshold[selected_type][0] = min(threshold[selected_type][0], temp)
                threshold[selected_type][1] = max(threshold[selected_type][1], temp)

                # Agli array usati per calcolare media e deviazione vengono appesi i risultati
                if temp not in threshold[selected_type][2]:
                    threshold[selected_type][2].append(temp)
                if temp not in threshold[selected_type][3]:
                    threshold[selected_type][3].append(temp)
            # Se sono diversi, restituisce il threshold l'iterazione
            else:
                return threshold


def prepare_normals(normals_path):
    normals_df = pd.read_csv(normals_path, delimiter='\t', low_memory=False)

    ts = pd.read_csv('/home/musimathicslab/Detection-signature-cancer/Normals2/normals_status.csv', delimiter='\t',
                     low_memory=False)

    ts_i = pd.read_csv('/home/musimathicslab/Detection-signature-cancer/Normals2/normals_index.csv', delimiter='\t',
                       low_memory=False)

    return normals_df, ts['TUMOR_STATUS'], ts_i['index']


def indexes_cancer_type(cancer_type):
    indexes = {}

    for i in range(len(cancer_type)):
        if cancer_type.iloc[i].values[0] not in indexes:
            indexes[cancer_type.iloc[i].values[0]] = i
        else:
            continue

    return indexes


def random_index(indexes, k):
    items = list(indexes.items())

    results = {}

    for i in range(len(items) - 1):
        name, value = items[i]
        next_value = items[i + 1][1]

        random_v = []
        for j in range(k):
            random_v.append(random.randint(value, next_value))

        results[name] = random_v
    return results


def siamese_network_normals(siamese_path, risultati_siamese, dataset_genes, model,
                    input_shape, genes_len, cancer_type, siamese_variants, normals_path,
                    normals_max_epsilon, normals_param_epsilon):
    # siamese_model = get_siamese_model(input_shape, model)
    with custom_object_scope({'initialize_bias': initialize_bias}):
        siamese_model = load_model(siamese_path + "siamese_model.keras", safe_mode=False,
                                   custom_objects={'initialize_bias': initialize_bias})

    siamese_model.summary()

    # Pre-Processing dataset for siamese network
    dataset_genes = pd.concat([dataset_genes, cancer_type], axis=1)

    dataset_genes = dataset_genes.drop('CANCER_TYPE', axis=1)

    indexes = indexes_cancer_type(cancer_type)

    normals_df, normals_ts, normals_index = prepare_normals(normals_path)

    threshold_general = {}

    valori_unici = cancer_type['CANCER_TYPE'].unique()
    for unico in valori_unici:
        threshold = calc_threshold(siamese_model, dataset_genes, cancer_type, unico, indexes[unico])

        media = np.mean(threshold[unico][2])
        dev = np.std(threshold[unico][3])
        threshold[unico][2] = None
        threshold[unico][3] = None
        threshold[unico][2] = media
        threshold[unico][3] = dev
        threshold_general.update(threshold)
        print(f"Threshold generale: {threshold_general}")
        with open(f'/home/musimathicslab/Detection-signature-cancer/Threshold/threshold_{unico}', 'w') as file:
            for chiave, valore in threshold.items():
                file.write(f"{chiave}: {valore[0]}, {valore[1]}, {valore[2]}, {valore[3]}\n")

    with open(f'/home/musimathicslab/Detection-signature-cancer/threshold.txt', 'w') as file:
        for chiave, valore in threshold_general.items():
            file.write(f"{chiave}: {valore[0]}, {valore[1]}, {valore[2]}, {valore[3]}\n")

    sample_count = 3
    rand_ind = random_index(indexes, sample_count)

    over_list = []
    out_list = []

    for k, v in rand_ind.items():
        for j in normals_df.index:
            count = 0
            input_normal = tf.convert_to_tensor(normals_df.iloc[j])
            input_normal = tf.expand_dims(input_normal, axis=0)

            for z in range(sample_count):

                input_sample = tf.convert_to_tensor(dataset_genes.iloc[v[z]])
                input_sample = tf.expand_dims(input_sample, axis=0)

                temp = siamese_model.predict([input_normal, input_sample], verbose=False)[0][0]

                for chiave, valore in threshold_general.items():
                    if normals_param_epsilon:
                        epsilon = 0.03
                    elif normals_max_epsilon:
                        epsilon = valore[3]

                    if normals_ts[j] == '0':
                        if (valore[2] - epsilon) <= temp <= (valore[2] + epsilon) and k == chiave:
                            over_list.append({
                                                 f"Paziente: {normals_index[j]}, Predetto: {temp}, Possibile: {chiave}, Threshold compreso: minimo: {valore[2] - valore[3]} - Massimo:{valore[2] + valore[3]}"})
                            count += 1
            if (count >= sample_count * 0.66):
                out_list.append({f"Paziente: {normals_index[j]}, Possibile: {k}"})

    if normals_param_epsilon:
        with open('/home/musimathicslab/Detection-signature-cancer/Results_Comparison/Over_Comparison/results_comparison_epsilon_003.txt', 'w') as file:
            for over in over_list:
                file.write(f"{over}\n")

        with open('/home/musimathicslab/Detection-signature-cancer/Results_Comparison/Over_Percentage/results_over_percentage_epsilon_003.txt', 'w') as file:
            for over in out_list:
                file.write(f"{over}\n")

    elif normals_max_epsilon:
        with open('/home/musimathicslab/Detection-signature-cancer/Results_Comparison/Over_Comparison/results_comparison_max_epsilon.txt', 'w') as file:
            for over in over_list:
                file.write(f"{over}\n")

        with open('/home/musimathicslab/Detection-signature-cancer/Results_Comparison/Over_Percentage/results_over_percentage_max_epsilon.txt', 'w') as file:
            for over in out_list:
                file.write(f"{over}\n")