import pandas as pd

input_file = '../../Dataset/output_trasposto.csv'
output_file = '../../Dataset/output_trasposto_copy.csv'

# Leggi il CSV utilizzando pandas, ignorando la prima colonna come indice
df = pd.read_csv(input_file, delimiter=';', header=None, low_memory=False)
# Calcola la deviazione standard per ciascuna colonna a partire dalla seconda colonna
std_dev = df.iloc[1:, 1:].std()

# Seleziona le colonne con deviazione standard maggiore o uguale a 0.8
selected_columns = std_dev[std_dev >= 0.8].index

# Aggiungi la prima colonna al DataFrame filtrato
selected_columns = [0] + selected_columns

# Crea un nuovo DataFrame con solo le colonne selezionate
df_filtered = df[selected_columns]

# Scrivi il DataFrame filtrato nel file di output
df_filtered.to_csv(output_file, index=False, sep=';')
