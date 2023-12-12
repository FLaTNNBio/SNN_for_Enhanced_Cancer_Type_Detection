import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def create_support_set(x_train, y_train, classes,n):
    x_support = []
    y_support = []
    for c in classes:
        indices = np.where(y_train == c)
        for i in range(0,n):
            x_support.append(x_train[indices[0][i]])
            y_support.append(y_train[indices[0][i]])
    return np.array(x_support),np.array(y_support)


def eval_dnn(x_test, y_test, y_test_ohe, classes, model):
    results = pd.DataFrame(columns=["Cancer", "Loss", "Accuracy", "Precision", "Recall", "AUC", "F1 Score"])
    for cancer in classes:
        indices = np.where(y_test == cancer)
        x_test_eval = x_test[indices]
        y_test_eval = y_test_ohe[indices]
        (loss, accuracy, precision, recall, auc) = model.evaluate(x_test_eval,y_test_eval)
        f1_score = 2*((precision*recall)/(precision+recall+K.epsilon()))
        new_row = {"Cancer": cancer, "Loss": loss, "Accuracy": accuracy, "Precision": precision,
                   "Recall": recall, "AUC": auc, "F1 Score": f1_score}
        results = pd.concat([results,pd.DataFrame.from_dict([new_row])])
    return results


def pre_processing(dataset_path, encoded_path, data_encoded):
    dataset_df = pd.read_csv(dataset_path, delimiter=';', low_memory=False)

    dataset_df.head()
    dataset_df = dataset_df[dataset_df['CANCER_TYPE'] != '[Not Available]']

    grouped = dataset_df.groupby('CANCER_TYPE').filter(lambda x: len(x) >= 30)
    dataset_df = dataset_df[dataset_df['CANCER_TYPE'].isin(grouped['CANCER_TYPE'].unique())]

    category_counts = dataset_df.groupby('CANCER_TYPE').size().reset_index(name='counts')
    category_counts['percentage'] = (category_counts['counts'] / len(dataset_df) * 100).round(2)
    print(category_counts)
    print("\n")

    average_percentage = category_counts['percentage'].mean()
    print(f"Media delle percentuali di campioni per classe: {average_percentage}%")

    constant_columns = [col for col in dataset_df.columns if dataset_df[col].nunique() == 1]
    dataset_df = dataset_df.drop(columns=constant_columns)

    # Commentiamo questa parte perchè altrimenti elimina alcuni cancri che hanno delle colonne con valori NaN
    '''dataset_df = dataset_df.dropna()
    dataset_df = dataset_df.reset_index(drop=True)'''

    classes = pd.unique(dataset_df["CANCER_TYPE"])
    n_classes = len(pd.unique(classes))
    del dataset_df["index"]

    le = LabelBinarizer()

    y_df = dataset_df["CANCER_TYPE"]
    y_df.head()
    del dataset_df["CANCER_TYPE"]

    y = y_df.to_numpy()

    if not data_encoded:
        object_cols = list(dataset_df.select_dtypes(include='object'))
        other_cols = list(dataset_df.select_dtypes(exclude='object'))
        ohe_encoded_data = pd.get_dummies(dataset_df, columns=object_cols, drop_first=False)

        encoded_data = pd.concat([ohe_encoded_data, dataset_df[other_cols]], axis=1)

        encoded_data.to_csv("encoded-dataset.csv", index=False, sep=';')
        print(np.array(encoded_data).shape)
    else:
        encoded_data = pd.read_csv(encoded_path, delimiter=';', low_memory=False)

    return encoded_data, n_classes, classes, le, y


def classification_model(encoded_data, n_classes, classes, le, y):
    # test_size: per default è 0.25 --> 25% test, 75% train

    # random_state: per default è su 'None' cioè il dataset ogni volta viene suddiviso in maniera casuale
    # e non in maniera deterministica (cioè, se eseguiamo il codice più volte con lo stesso valore di random_state,
    # otterremo sempre la stessa suddivisione dei dati tra training e testing).

    # stratify=y: indica che si desidera mantenere la stessa distribuzione delle classi nel training e nel testing.
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(encoded_data), y,  stratify=y, test_size=0.25, random_state=42)

    y_train_ohe = le.fit_transform(y_train)

    inputs = Input(x_train[0].shape)
    input_shape = x_train[0].shape
    input1 = Input(input_shape)

    x_support, y_support = create_support_set(x_train, y_train, classes, 2)

    input_layer = Dense(np.array(encoded_data).shape[1], activation='relu', input_shape=x_train[0].shape)(input1)

    hidden_layers = Dense(750, activation='relu')(input_layer)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Dense(750, activation='relu')(hidden_layers)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Dense(750, activation='relu')(hidden_layers)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Dense(750, activation='relu')(hidden_layers)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Dense(750, activation='relu')(hidden_layers)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Dropout(0.1)(hidden_layers)
    classifier = Dense(n_classes, activation='softmax')(hidden_layers)

    model = Model(inputs=input_layer, outputs=classifier)
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(x_train, y_train_ohe, batch_size=64, epochs=100, validation_split=0.2,
                        callbacks=[early_stopping])

    y_test_ohe = le.fit_transform(y_test)

    model.evaluate(x_test, y_test_ohe)
    model.save('model.h5')

    dnn_results = eval_dnn(x_test, y_test, y_test_ohe, classes, model)
    print("DNN Results:")
    print(dnn_results)

    dnn_results.to_csv("risultati_testing.csv", index=False, sep=';')

    return model, x_support, y_support, x_train, y_train, input_shape, x_test, y_test
