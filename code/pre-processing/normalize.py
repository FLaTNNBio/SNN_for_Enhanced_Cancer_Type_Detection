import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Carica il CSV
csv_path = '/home/alberto/Scrivania/acc_tcga/data_rppa.csv'
df = pd.read_csv(csv_path, sep=';')

# Specifica le colonne da normalizzare (dalla terza in poi)
columns_to_normalize = df.columns[1:]

# Verifica se ci sono almeno due righe per normalizzare
if len(df) > 1:
    # Crea un oggetto MinMaxScaler
    scaler = MinMaxScaler()

    # Normalizza tutte le colonne dalla terza in poi
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # Salva il CSV normalizzato
    output_csv_path = '/home/alberto/Scrivania/acc_tcga/data_rppa.csv'
    df.to_csv(output_csv_path, index=False)

    print(f'Dati normalizzati e salvati in: {output_csv_path}')
else:
    print('Il DataFrame ha meno di due righe. Impossibile normalizzare.')
