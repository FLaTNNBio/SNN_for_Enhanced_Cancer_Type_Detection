from classification import *
from siamese_network import *
from classification_normals import *
from siamese_with_normals import *
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    only_variant = False
    data_encoded = False
    classification = False
    classification_normals = False
    siamese_net = False
    siamese_normals = True
    normals_max_epsilon = False
    normals_param_epsilon = True
    siamese_variants = False

    ####################################################################################################################
    # DATA
    dataset_path = (
        "/home/musimathicslab/Detection-signature-cancer/Dataset/data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005_dataPatient_mutations_and_variants2.csv")

    encoded_path = "/home/musimathicslab/Detection-signature-cancer/code/models/0030/classification/espressione_genomica_con_varianti_2LAYER/encoded-dataset.csv"

    ####################################################################################################################

    # CLASSIFICATION
    model_path = "/home/musimathicslab/Detection-signature-cancer/code/models/0005/classification/espressione_genomica_con_varianti_2LAYER/"

    risultati_classification = ("/home/musimathicslab/Detection-signature-cancer/code/risultati/classification/0005/"
                                "classification_espressione_genomica_con_varianti_2LAYER.csv")

    ####################################################################################################################

    # SIAMESE
    siamese_path = "/home/musimathicslab/Detection-signature-cancer/code/models/0005/siamese/espressione_genomica_con_varianti_2LAYER/"

    risultati_siamese = "/home/musimathicslab/Detection-signature-cancer/code/risultati/siamese/0005/siamese_espressione_genomica_con_varianti_2LAYER.txt"

    ####################################################################################################################

    # NORMALS
    normals_path = "/home/musimathicslab/Detection-signature-cancer/Normals2/output_variants_cna_trasposto_normalizzato_deviazione.csv"

    ####################################################################################################################

    if classification_normals:
        encoded_data, normals_df, weights, n_classes, classes, le, y = pre_processing_normals(normals_path, encoded_path,
                                                                                              data_encoded, only_variant)
    else:
        encoded_data, dataset_df, weights, n_classes, classes, le, y = pre_processing(dataset_path, encoded_path,
                                                                                          data_encoded, only_variant)

    if classification:
        print("Start classification:\n")
        model = classification_model(model_path, risultati_classification, encoded_data, weights, n_classes,
                                     classes, le, y)
        print('Ended classification:\n')
    elif classification_normals:
        print("Start classification with normals:\n")
        model = classification_normals_model(model_path, risultati_classification, encoded_data, weights, n_classes,
                                             classes, le, y)
        print('Ended classification with normals:\n')

    genes_len = dataset_df.shape[1]
    input_shape = (genes_len, 1)

    cancer_type = pd.DataFrame(y, columns=['CANCER_TYPE'])

    if siamese_net:
        print('Start siamese network:\n')
        if not classification:
            model = load_model(model_path + "model.keras", custom_objects={'loss': weighted_categorical_crossentropy})

        siamese_network(siamese_path, risultati_siamese, dataset_df, model,
                            input_shape, genes_len, cancer_type, siamese_variants)
    elif siamese_normals:
        print('Start siamese network with normals:\n')
        model = load_model(model_path + "model.keras", custom_objects={'loss': weighted_categorical_crossentropy})
        siamese_network_normals(siamese_path, risultati_siamese, dataset_df, model,
                        input_shape, genes_len, cancer_type, siamese_variants,
                        normals_path, normals_max_epsilon, normals_param_epsilon)
