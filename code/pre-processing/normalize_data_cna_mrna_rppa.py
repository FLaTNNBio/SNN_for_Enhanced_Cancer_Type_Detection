import os
import pandas as pd
import numpy as np


def normalizza_e_concatena(input_csv_path, output_csv_path):
    # Carica il CSV in un DataFrame
    df = pd.read_csv(input_csv_path, sep=';')

    # Seleziona le colonne numeriche a partire dalla x colonna
    colonne_numeriche = df.iloc[:, 1:].select_dtypes(include=[np.number])

    # Trova i valori minimi e massimi dalle colonne selezionate
    min_val = colonne_numeriche.min().min()
    max_val = colonne_numeriche.max().max()

    # Normalizza tutte le colonne
    normalized_columns = (colonne_numeriche - min_val) / (max_val - min_val)

    # Carica il file originale per ottenere le prime due colonne
    df_original = pd.read_csv(input_csv_path, sep=';')

    # Concatena le prime due colonne con i dati normalizzati
    df_out = pd.concat([df_original.iloc[:, :1], normalized_columns], axis=1)

    # Salva il CSV normalizzato e concatenato
    df_out.to_csv(output_csv_path, index=False, sep=';')

    print(f'Dati normalizzati e concatenati salvati in: {output_csv_path}')


def elabora_sottocartelle(directory_principale):
    for root, dirs, files in os.walk(directory_principale):
        if 'data_rppa.csv' in files:
            input_csv_path = os.path.join(root, 'data_rppa.csv')
            output_csv_path = os.path.join(root, 'data_rppa.csv')
            normalizza_e_concatena(input_csv_path, output_csv_path)


# Specifica la directory principale in cui eseguire le operazioni
directory_principale = "/home/alberto/Scrivania/Dataset_2"
elabora_sottocartelle(directory_principale)