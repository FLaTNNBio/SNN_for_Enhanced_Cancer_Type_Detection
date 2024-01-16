import os
import csv


def cerca_gene(directory):
    # Trova tutti i file data_mutations.csv nelle sottocartelle della directory
    files = [os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files if
             file.endswith('data_mutations.csv')]

    for file_path in files:
        with open(file_path, 'r') as csv_file:
            # Leggi il file CSV
            csv_reader = csv.DictReader(csv_file, delimiter=';')

            # Controlla se la colonna NCBI_Build esiste
            if 'NCBI_Build' in csv_reader.fieldnames:
                # Verifica se almeno una riga ha il valore GRCh37 nella colonna NCBI_Build
                for row in csv_reader:
                    if 'NCBI_Build' in row and row['NCBI_Build'] == 'GRCh37':
                        print("--------------------------------------------")
                        break
                else:
                    print(f"File: {file_path} - Nessuna riga con NCBI_Build = GRCh37")
            else:
                print(f"File: {file_path} - Colonna NCBI_Build non trovata")


cerca_gene('/media/alberto/DATA/Cancer dataset/dataset')
