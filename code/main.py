from classification import *
from siamese_network import *
from tensorflow.keras.models import load_model


if __name__ == "__main__":
    only_variant = False
    data_encoded = False
    classification = True
    siamese_net = True

    dataset_path = ("/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/dataset/data_mrna/"
                    "deviazione_standard_dataPatient/"
                    "data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0030_dataPatient.csv")

    encoded_path = ("/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/models/0030/espressione_genomica/"
                    "encoded-dataset.csv")

    model_path = "/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/models/0030/espressione_genomica/"

    risultati_classification = ("/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/risultati/"
                                "classification/0030/classification_espressione_genomica.csv")

    risultati_siamese = ("/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/risultati/siamese/0030/"
                         "siamese_espressione_genomica.txt")

    encoded_data, dataset_df, weights, n_classes, classes, le, y = (
        pre_processing(dataset_path, encoded_path, data_encoded, only_variant))

    if classification:
        model = classification_model(model_path, risultati_classification, encoded_data, weights, n_classes, classes, le, y)

    if siamese_net:
        if not classification:
            model = load_model(model_path + "model.keras", custom_objects={'loss': weighted_categorical_crossentropy})

        genes_len = dataset_df.shape[1]
        input_shape = (genes_len, 1)

        cancer_type = pd.DataFrame(y, columns=['CANCER_TYPE'])
        siamese_network(risultati_siamese, dataset_df, model, input_shape, genes_len, cancer_type)
