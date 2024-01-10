import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def create_support_set(x_train, y_train, classes, n):
    x_support = []
    y_support = []
    for c in classes:
        indices = np.where(y_train == c)
        for i in range(0, n):
            x_support.append(x_train[indices[0][i]])
            y_support.append(y_train[indices[0][i]])
    return np.array(x_support), np.array(y_support)


def eval_dnn(x_test, y_test, y_test_ohe, classes, model):
    results = pd.DataFrame(columns=["Cancer", "Loss", "Accuracy", "Precision", "Recall", "AUC", "F1 Score"])
    for cancer in classes:
        indices = np.where(y_test == cancer)
        x_test_eval = x_test[indices]
        y_test_eval = y_test_ohe[indices]
        (loss, accuracy, precision, recall, auc) = model.evaluate(x_test_eval, y_test_eval)
        f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
        new_row = {"Cancer": cancer, "Loss": loss, "Accuracy": accuracy, "Precision": precision,
                   "Recall": recall, "AUC": auc, "F1 Score": f1_score}
        results = pd.concat([results, pd.DataFrame.from_dict([new_row])])
    return results


def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def pre_processing(dataset_path, encoded_path, data_encoded):
    dataset_df = pd.read_csv(dataset_path, delimiter=';', low_memory=False)
    dataset_df = dataset_df[dataset_df['CANCER_TYPE'] != '[Not Available]']

    # grouped = dataset_df.groupby('CANCER_TYPE').filter(lambda x: len(x) >= 30)
    # dataset_df = dataset_df[dataset_df['CANCER_TYPE'].isin(grouped['CANCER_TYPE'].unique())]

    dataset_df = dataset_df.dropna()
    dataset_df = dataset_df.reset_index(drop=True)

    category_counts = dataset_df.groupby('CANCER_TYPE').size().reset_index(name='counts')
    category_counts['percentage'] = (category_counts['counts'] / len(dataset_df) * 100).round(2)
    print(category_counts)
    print("\n")

    average_percentage = category_counts['percentage'].mean()
    print(f"Media delle percentuali di campioni per classe: {average_percentage} %")

    constant_columns = [col for col in dataset_df.columns if dataset_df[col].nunique() == 1]
    dataset_df = dataset_df.drop(columns=constant_columns)

    ############################################################################################
    dataset_df = dataset_df.drop(columns=["SEX", "SAMPLE_TYPE", "ONCOTREE_CODE", "OS_STATUS",
                                          "AJCC_PATHOLOGIC_TUMOR_STAGE", "AGE"])
    ############################################################################################

    classes = pd.unique(dataset_df["CANCER_TYPE"])
    n_classes = len(pd.unique(classes))
    del dataset_df["index"]

    le = LabelBinarizer()

    y_df = dataset_df["CANCER_TYPE"]
    del dataset_df["CANCER_TYPE"]

    y = y_df.to_numpy()

    if not data_encoded:
        object_cols = list(dataset_df.select_dtypes(include='object'))
        other_cols = list(dataset_df.select_dtypes(exclude='object'))
        ohe_encoded_data = pd.get_dummies(dataset_df, columns=object_cols, drop_first=False)

        encoded_data = pd.concat([ohe_encoded_data, dataset_df[other_cols]], axis=1)

        encoded_data.to_csv(encoded_path, index=False, sep=';')
        print(np.array(encoded_data).shape)
    else:
        encoded_data = pd.read_csv(encoded_path, delimiter=';', low_memory=False)

    ratio_cancer = dataset_df.shape[0] / category_counts['counts']
    weights = np.array(ratio_cancer)

    return encoded_data, dataset_df, category_counts, weights, n_classes, classes, le, y


def created_model(input_shape, n_classes):
    input_layer = Input(shape=input_shape)
    conv1 = Conv1D(filters=256, kernel_size=50, strides=50, activation='relu', padding='same')(input_layer)
    conv2 = Conv1D(filters=128, kernel_size=10, strides=1, activation='relu', padding='same')(conv1)
    maxpool1 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='same')(maxpool1)
    maxpool2 = MaxPooling1D(pool_size=2)(conv3)

    flatten = Flatten()(maxpool2)
    output = Dense(n_classes, activation='softmax')(flatten)

    return output, input_layer


def classification_model(encoded_data, weights, n_classes, classes, le, y):
    # test_size: per default è 0.25 --> 25% test, 75% train

    # random_state: per default è su 'None' cioè il dataset ogni volta viene suddiviso in maniera casuale
    # e non in maniera deterministica (cioè, se eseguiamo il codice più volte con lo stesso valore di random_state,
    # otterremo sempre la stessa suddivisione dei dati tra training e testing).

    # stratify=y: indica che si desidera mantenere la stessa distribuzione delle classi nel training e nel testing.
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(encoded_data), y, stratify=y, test_size=0.25, random_state=42)

    y_train_ohe = le.fit_transform(y_train)

    x_train = np.asarray(x_train).astype('float32')
    y_train_ohe = np.asarray(y_train_ohe).astype('float32')

    input_shape = (x_train[0].shape[0], 1)

    model_out, input_layer = created_model(input_shape, n_classes)

    model = Model(inputs=input_layer, outputs=model_out)
    model.summary()
    model.compile(
        loss=weighted_categorical_crossentropy(weights),
        optimizer="adam",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    history = model.fit(x_train, y_train_ohe, batch_size=64, epochs=100, validation_split=0.2)

    y_test_ohe = le.fit_transform(y_test)

    x_test = np.asarray(x_test).astype('float32')
    y_test_ohe = np.asarray(y_test_ohe).astype('float32')

    model.evaluate(x_test, y_test_ohe)

    # Salvataggio modello
    model.save('/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/model.keras')

    dnn_results = eval_dnn(x_test, y_test, y_test_ohe, classes, model)
    print("DNN Results:")
    print(dnn_results)

    dnn_results.to_csv("/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/risultati_testing.csv",
                       index=False, sep=';')

    return model
