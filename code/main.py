from classification import *
from siamese_network import *


if __name__ == "__main__":
    dataset_path = ("/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/Dataset/data_mrna/"
                    "data_mrna_v2_seq_rsem_trasposto_normalizzato_dataPatient.csv")

    encoded_data, n_classes, classes, le, y = pre_processing(dataset_path)

    model, x_support, y_support, x_train, y_train, input_shape, x_test, y_test = (
        classification_model(encoded_data, n_classes, classes, le, y))

    siamese_network(model, classes, x_support, y_support, x_train, y_train, input_shape, x_test, y_test)
