import pandas as pd

input_file = '../../Dataset/data_mrna/data_mrna_v2_seq_rsem_trasposto_normalizzato.csv'
output_file = '../../Dataset/data_mrna/data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0002.csv'

# Leggi il CSV utilizzando pandas, ignorando la prima colonna come indice
df = pd.read_csv(input_file, delimiter=';', header=None, low_memory=False)

# Calcola la deviazione standard per ciascuna colonna a partire dalla seconda colonna
std_dev = df.iloc[1:, 1:].std()

# Seleziona le colonne con deviazione standard maggiore o uguale a 0.8
selected_columns = std_dev[std_dev >= 0.002].index
print(selected_columns)

# Aggiungi la prima colonna al DataFrame filtrato
selected_columns = [0] + selected_columns

# Crea un nuovo DataFrame con solo le colonne selezionate
df_filtered = df[selected_columns]

# Scrivi il DataFrame filtrato nel file di output
df_filtered.to_csv(output_file, index=False, sep=';')
