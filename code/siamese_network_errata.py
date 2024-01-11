import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.models import Model
from classification import weighted_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping


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


def siamese_network(model, classes, weights, x_support, y_support, x_train, y_train,
                    input_shape, x_test, y_test):
    print("\nCreating couples...")
    x_train_left, x_train_right, y_train_set = create_couples(x_support, y_support, x_train, y_train)

    x_train_left = np.asarray(x_train_left).astype('float32').reshape(-1, x_train[0].shape[0], 1)
    x_train_right = np.asarray(x_train_right).astype('float32').reshape(-1, x_train[0].shape[0], 1)
    y_train_set = np.asarray(y_train_set).astype('float32')

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
    siamese_model.compile(loss=weighted_categorical_crossentropy(weights),
                          optimizer="adam",
                          metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC()])

    print("Fit in progress...")
    x_train_left = np.asarray(x_train_left).astype('float32')
    x_train_right = np.asarray(x_train_right).astype('float32')
    y_train_set = np.asarray(y_train_set).astype('float32')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    history_s = siamese_model.fit([x_train_left, x_train_right], y=y_train_set, batch_size=64, epochs=40,
                                  validation_split=0.2, callbacks=[early_stopping])

    x_test_left, x_test_right, y_test_set = create_couples(x_support, y_support, x_test, y_test)

    print("Evaluate in progress...")
    x_test_left = np.asarray(x_test_left).astype('float32')
    x_test_right = np.asarray(x_test_right).astype('float32')
    y_test_set = np.asarray(y_test_set).astype('float32')

    x_test_left = x_test_left.reshape(x_test_left.shape + (1,))
    x_test_right = x_test_right.reshape(x_test_right.shape + (1,))
    y_test_set = y_test_set.reshape(y_test_set.shape + (1,))

    (loss, accuracy, precision, recall, auc) = siamese_model.evaluate([x_test_left, x_test_right], y_test_set)

    results = eval_siamese_model(loss, accuracy, precision, recall, auc, classes)
    print("\nSiamese Results:")
    print(results)

    results.to_csv("/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/model_0030_variants/"
                   "siamese/risultati_testing_siamese.csv",
                   index=False, sep=';')
