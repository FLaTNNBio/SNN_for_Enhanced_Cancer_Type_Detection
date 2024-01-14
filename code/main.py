from classification import *
from siamese_network import *
from tensorflow.keras.models import load_model


if __name__ == "__main__":
    data_encoded = True
    classification = False
    siamese_net = True

    dataset_path = ("/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/Dataset/data_mrna/"
                    "deviazione_standard_dataPatient/"
                    "data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0030_dataPatient.csv")

    encoded_path = "/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/encoded-dataset.csv"

    model_path = "/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/model.keras"

    encoded_data, dataset_df, weights, n_classes, classes, le, y = (
        pre_processing(dataset_path, encoded_path, data_encoded))

    if classification:
        model = classification_model(encoded_data, weights, n_classes, classes, le, y)

    if siamese_net:
        if not classification:
            model = load_model(model_path, custom_objects={'loss': weighted_categorical_crossentropy})

        genes_len = dataset_df.shape[1]
        input_shape = (genes_len, 1)
        print(genes_len)

        cancer_type = pd.DataFrame(y, columns=['CANCER_TYPE'])
        siamese_network(dataset_df, model, input_shape, genes_len, cancer_type)
