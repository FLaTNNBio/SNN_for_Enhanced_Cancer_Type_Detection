import os
import pandas as pd


def elimina_colonna(csv_path):
    df = pd.read_csv(csv_path, delimiter=';')

    # Verifica se la colonna "Entrez_Gene_Id" esiste nel DataFrame
    if "Entrez_Gene_Id" in df.columns:
        df = df.drop(columns=["Entrez_Gene_Id"])

        df.to_csv(csv_path, index=False, sep=';')
        print(f"Colonna 'Entrez_Gene_Id' eliminata da {csv_path}")
    else:
        print(f"La colonna 'Entrez_Gene_Id' non esiste in {csv_path}")


def main(cartella_principale):
    # Trova tutti i file CSV nelle sottocartelle di cartella_principale
    for cartella, _, files in os.walk(cartella_principale):
        for file in files:
            # Verifica se il file è un CSV di nome 'data_mrna_v2_seq_rsem'
            if file.startswith('data_mrna_v2_seq_rsem') and file.endswith('.csv'):
                csv_path = os.path.join(cartella, file)
                elimina_colonna(csv_path)


main('/home/alberto/Scrivania/Dataset_clean')
