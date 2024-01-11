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

    encoded_path = ("/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/model_0030_mutations/"
                    "encoded-dataset.csv")

    model_path = ("/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/model_0030_mutations/classification/"
                  "model.keras")

    encoded_data, dataset_df, category_counts, weights, n_classes, classes, le, y = (
        pre_processing(dataset_path, encoded_path, data_encoded))

    if classification:
        model = classification_model(encoded_data, weights, n_classes, classes, le, y)

    if siamese_net:
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(encoded_data), y, stratify=y, test_size=0.25, random_state=42)

        x_support, y_support = create_support_set(x_train, y_train, classes, 2)

        '''input_shape = (x_train[0].shape[0], 1)'''

        if not classification:
            model = load_model(model_path, custom_objects={'loss': weighted_categorical_crossentropy})

        '''siamese_network(model, classes, weights, x_support, y_support, x_train,
                        y_train, input_shape, x_test, y_test)'''

        genes_len = dataset_df.shape[1]
        input_shape = (genes_len, 1)

        cancer_type = pd.DataFrame(y, columns=['CANCER_TYPE'])
        siamese_network_test(dataset_df, model, input_shape, genes_len, classes, n_classes, cancer_type)
