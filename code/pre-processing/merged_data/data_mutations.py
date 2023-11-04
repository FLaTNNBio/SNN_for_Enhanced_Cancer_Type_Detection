import os
import pandas as pd


def leggi_csv(cartella):
    dati = []

    for root, dirs, files in os.walk(cartella):
        for file in files:
            if file.endswith("data_mutations.csv"):
                percorso_file = os.path.join(root, file)
                df = pd.read_csv(percorso_file, delimiter=';')
                dati.append(df)
    return dati


cartella_principale = "/home/alberto/Scrivania/Dataset (completo)"
dati_concatenati = pd.concat(leggi_csv(cartella_principale), ignore_index=True)
dati_senza_duplicati = dati_concatenati.drop_duplicates()

# Salva il dataframe in un nuovo file CSV
percorso_output = "/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/Dataset/data_mutations.csv"
dati_senza_duplicati.to_csv(percorso_output, index=False, sep=';')
