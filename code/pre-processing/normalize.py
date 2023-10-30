import pandas as pd

# Carica il CSV in un DataFrame
df = pd.read_csv('/home/alberto/Scrivania/Dataset/sarc_tcga_pub/data_cna.csv', sep=';')

# Seleziona la seconda riga in poi e la terza colonna in poi
selected_data = df.iloc[0:, 2:]

# Find the minimum and maximum values from both columns
min_val = selected_data.min().min()
max_val = selected_data.max().max()

# Normalizzare tutte le colonne
normalized_columns = (selected_data - min_val) / (max_val - min_val)

# Print the DataFrame with normalized columns
print(normalized_columns)

# Salva il CSV normalizzato
df_out = pd.DataFrame(normalized_columns)
output_csv_path = '/home/alberto/Scrivania/Dataset/sarc_tcga_pub/data_cnaa.csv'
df_out.to_csv(output_csv_path, index=False, sep=';')

print(f'Dati normalizzati e salvati in: {output_csv_path}')
