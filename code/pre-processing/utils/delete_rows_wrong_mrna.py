import os
import pandas as pd

cartella_dati_input = "/home/alberto/Scrivania/Dataset_clean"
cartella_dati_output = "/home/alberto/Scrivania/prova"

# Crea la cartella di output se non esiste
if not os.path.exists(cartella_dati_output):
    os.makedirs(cartella_dati_output)

# Itera sui numeri da 1 a 67
for numero in range(1, 68):
    nome_file_input = f"data_mrna_seq_v2_rsem_{numero}.csv"
    percorso_file_input = os.path.join(cartella_dati_input, nome_file_input)

    nome_file_output = f"data_mrna_v2_seq_rsem_{numero}.csv"
    percorso_file_output = os.path.join(cartella_dati_output, nome_file_output)

    # Controlla se il file di input esiste
    if os.path.exists(percorso_file_input):
        print(percorso_file_input)
        df = pd.read_csv(percorso_file_input, delimiter=';')

        # Elimina le righe con valori numerici nella colonna 'Hugo_Symbol'
        df = df[~df['Hugo_Symbol'].astype(str).str.isdigit()]

        df.to_csv(percorso_file_output, index=False, sep=';')

print("Operazioni completate con successo. File modificati salvati in", cartella_dati_output)
