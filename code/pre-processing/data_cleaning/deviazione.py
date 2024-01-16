import pandas as pd
import matplotlib.pyplot as plt

input_file = '../../dataset/data_mrna/data_mrna_v2_seq_rsem_trasposto_normalizzato.csv'
output_file = '../../dataset/data_mrna/data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0002.csv'

df = pd.read_csv(input_file, delimiter=';')

# Calcola la deviazione standard per ogni colonna (tranne la prima)
deviazioni_std = df.iloc[:, 1:].std()

# Filtra le colonne che superano la deviazione standard desiderata
colonne_da_salvare = deviazioni_std[deviazioni_std >= 0.00005].index.tolist()
colonne_da_salvare.insert(0, df.columns[0])  # Aggiungi la prima colonna

# Crea un nuovo DataFrame con le colonne selezionate
nuovo_df = df[colonne_da_salvare]

# Salva il nuovo DataFrame in un nuovo file CSV
nuovo_df.to_csv(output_file, index=False, sep=';')

print(f"Colonne salvate in {output_file}")
print(f"Numero finale di colonne: {nuovo_df.shape[1]}")


################################################################################
# Grafico
plt.scatter(deviazioni_std.index, deviazioni_std)
plt.title('Deviazione Standard per Colonna')
plt.xlabel('Colonne')
plt.ylabel('Deviazione Standard')
plt.xticks(rotation=90)
plt.show()
################################################################################


################################################################################
# Scrivi le deviazioni standard nel file di testo
deviation_output_file = "../../dataset/data_mrna/deviation.txt"

with open(deviation_output_file, 'w') as file:
    for column, deviation in zip(deviazioni_std.index, deviazioni_std.values):
        file.write(f'{column}: {deviation}\n')
################################################################################
