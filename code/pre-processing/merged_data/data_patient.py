import os
import pandas as pd


def concat_and_remove_duplicates(main_folder):
    # Lista per salvare i frame dei singoli CSV
    frames = []

    # Scansiona tutte le sottocartelle di main_folder
    for subdir, _, files in os.walk(main_folder):
        for file in files:
            # Verifica se il file è un CSV e ha il nome desiderato
            if file.endswith(".csv") and file.startswith("data_clinical_patient"):
                file_path = os.path.join(subdir, file)
                # Leggi il CSV e aggiungi il frame alla lista
                df = pd.read_csv(file_path, delimiter=';')
                frames.append(df)

    # Concatena tutti i frame in uno solo
    concatenated_df = pd.concat(frames, ignore_index=True)

    # Riempie le celle vuote con '[Not Available]'
    for column in concatenated_df.columns:
        concatenated_df[column].fillna('[Not Available]', inplace=True)

    # Elimina i duplicati basandosi sulla colonna 'SAMPLE_ID'
    deduplicated_df = concatenated_df.drop_duplicates(subset='SAMPLE_ID')

    # Salva il risultato in un nuovo file CSV
    deduplicated_df.to_csv(
        '/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/Dataset/data_clinical_patient.csv',
        index=False, sep=';')


# Specifica la cartella principale dove cercare i file CSV
main_folder = '/home/alberto/Scrivania/Dataset (completo)'

# Chiama la funzione per eseguire l'operazione
concat_and_remove_duplicates(main_folder)
