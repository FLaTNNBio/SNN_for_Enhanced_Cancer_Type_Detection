import os
import pandas as pd


def cerca_label_e_copia(input_csv):
    # Indice della riga contenente le label
    indice_label = 4

    # Lista delle label da cercare
    labels_da_cercare = ["SAMPLE_ID", "CANCER_TYPE", "SEX", "SAMPLE_TYPE", "SOMATIC_STATUS", "ONCOTREE_CODE",
                         "OS_STATUS", "AJCC_PATHOLOGIC_TUMOR_STAGE", "AGE"]

    # Leggi il CSV in un DataFrame pandas
    df = pd.read_csv(input_csv, header=indice_label, delimiter=';')

    # Dizionario per memorizzare le colonne corrispondenti alle label
    colonne_da_copiare = {}

    # Copia delle colonne corrispondenti alle label
    for label in labels_da_cercare:
        if label in df.columns:
            colonne_da_copiare[label] = df[label]

    # Creazione del DataFrame risultante
    df_risultante = pd.DataFrame(colonne_da_copiare)

    return df_risultante


def elabora_sottocartelle(directory_principale):
    for root, dirs, files in os.walk(directory_principale):
        if 'data_clinical_patient.csv' in files and 'data_clinical_sample.csv' in files:
            input_csv_patient = os.path.join(root, 'data_clinical_patient.csv')
            input_csv_sample = os.path.join(root, 'data_clinical_sample.csv')

            df_patient = cerca_label_e_copia(input_csv_patient)
            df_sample = cerca_label_e_copia(input_csv_sample)

            df_concatenato = pd.concat([df_patient, df_sample], axis=1)

            labels_ordine = ["SAMPLE_ID", "CANCER_TYPE", "SEX", "SAMPLE_TYPE", "SOMATIC_STATUS", "ONCOTREE_CODE",
                             "OS_STATUS", "AJCC_PATHOLOGIC_TUMOR_STAGE", "AGE"]

            labels_presenti = [label for label in labels_ordine if label in df_concatenato.columns]

            if len(labels_presenti) < len(labels_ordine):
                label_mancanti = set(labels_ordine) - set(labels_presenti)
                print(f"Avviso: Alcune label non trovate nel DataFrame concatenato: {label_mancanti}")

            df_risultante = df_concatenato[labels_presenti]

            output_path = os.path.join(root, 'data.csv')
            df_risultante.to_csv(output_path, sep=';', index=False)
            print(f"Operazione completata per {root}. Output salvato in {output_path}.\n\n")


# Specifica la directory principale in cui eseguire le operazioni
directory_principale = "/home/alberto/Scrivania/Dataset_2"
elabora_sottocartelle(directory_principale)
