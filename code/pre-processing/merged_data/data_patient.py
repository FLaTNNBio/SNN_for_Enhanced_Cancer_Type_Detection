import pandas as pd
import os
import glob


def merge_and_remove_duplicates(input_csv, output_csv, input_directory):
    # Leggi il primo CSV considerando l'header
    main_df = pd.read_csv(input_csv, delimiter=';')

    input_csv = 'data_clinical_patient.csv'
    # Cerca tutti i CSV con lo stesso nome nelle sottocartelle della directory principale
    pattern = os.path.join(input_directory, '**', os.path.basename(input_csv))
    additional_csv_files = glob.glob(pattern, recursive=True)

    # Leggi gli altri CSV senza considerare l'header, sostituendo i valori vuoti con [Not Available] e concatenali
    for csv_file in additional_csv_files:
        if csv_file != input_csv:
            try:
                df = pd.read_csv(csv_file, na_values='', keep_default_na=False, delimiter=';')
                df.fillna('[Not Available]', inplace=True)

                # Converte esplicitamente la colonna a un tipo di dato che può ospitare stringhe
                df['SAMPLE_ID'] = df['SAMPLE_ID'].astype(str)

                main_df = pd.concat([main_df, df], ignore_index=True)
            except pd.errors.ParserError as e:
                print(f"Error reading {csv_file}: {e}")

    # Rimuovi righe duplicate basate sulla colonna "SAMPLE_ID"
    main_df.drop_duplicates(subset='SAMPLE_ID', keep='first', inplace=True)

    # Salva il risultato in un nuovo CSV
    main_df.to_csv(output_csv, index=False, sep=';')


# Esempio di utilizzo:
merge_and_remove_duplicates('/home/alberto/Scrivania/acc_tcga/data_clinical_patient.csv',
                            '/home/alberto/Scrivania/Dataset/data_clinical_patient.csv',
                            '/home/alberto/Scrivania/Dataset (completo)')
