import numpy as np
import pandas as pd
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout


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


def eval_siamese_model(x_support, y_support, x_test, y_test, classes, siamese_model):
    results = pd.DataFrame(columns=["Cancer", "Loss", "Accuracy", "Precision", "Recall", "AUC", "F1 Score"])
    for cancer in classes:
        indices = np.where(y_test == cancer)
        x_test_eval = x_test[indices]
        y_test_eval = y_test[indices]
        x_test_left, x_test_right, y_test_set = create_couples(x_support, y_support, x_test_eval, y_test_eval)
        (loss, accuracy, precision, recall, auc) = siamese_model.evaluate([x_test_left, x_test_right], y_test_set)
        f1_score = 2*((precision*recall)/(precision+recall+K.epsilon()))
        new_row = {"Cancer": cancer, "Loss": loss, "Accuracy": accuracy, "Precision": precision,
                   "Recall": recall, "AUC": auc, "F1 Score": f1_score}
        results = pd.concat([results,pd.DataFrame.from_dict([new_row])])
    return results


def siamese_network(model, encoded_data, classes, x_support, y_support, x_train, y_train, input_shape, x_test, y_test):
    inputs1 = Input(np.array(encoded_data).shape[1])
    inputs2 = Input(np.array(encoded_data).shape[1])

    x_train_left, x_train_right, y_train_set = create_couples(x_support, y_support, x_train, y_train)

    left_model = Sequential(model.layers[:-1])
    right_model = Sequential(model.layers[:-1])

    for layer in left_model.layers:
        layer.trainable = False
    for layer in right_model.layers:
        layer.trainable = False

    left_input = Input(shape=input_shape, name='left_input')
    right_input = Input(shape=input_shape, name='right_input')

    left_output = left_model(left_input)
    right_output = right_model(right_input)

    merged_output = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True)))(
        [left_output, right_output])

    merged_output = Dense(320, activation="relu")(merged_output)
    merged_output = Dropout(0.1)(merged_output)
    merged_output = BatchNormalization()(merged_output)
    merged_output = Dense(320, activation="relu")(merged_output)
    merged_output = Dropout(0.1)(merged_output)
    merged_output = BatchNormalization()(merged_output)
    merged_output = Dense(320, activation="relu")(merged_output)
    merged_output = Dropout(0.1)(merged_output)
    merged_output = BatchNormalization()(merged_output)

    final_output = Dense(1, activation='sigmoid')(merged_output)

    siamese_model = Model(inputs=[left_input, right_input], outputs=final_output)

    siamese_model.compile(loss='binary_crossentropy', optimizer="adam",
                          metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC()])

    history_s = siamese_model.fit([x_train_left, x_train_right], y=y_train_set, batch_size=64, epochs=40,
                                  validation_split=0.2)

    x_test_left, x_test_right, y_test_set = create_couples(x_support, y_support, x_test, y_test)

    (loss, accuracy, precision, recall, auc) = siamese_model.evaluate([x_test_left, x_test_right], y_test_set)

    results = eval_siamese_model(x_support, y_support, x_test, y_test, classes, siamese_model)
    print("Siamese Results:")
    print(results)
