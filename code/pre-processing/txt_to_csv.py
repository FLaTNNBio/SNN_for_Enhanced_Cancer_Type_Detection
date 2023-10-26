import os
import csv

input_directory = '/home/alberto/Scrivania/acc_tcga'
output_directory = '/home/alberto/Scrivania/acc_tcga'

# Assicurati che la cartella di output esista, altrimenti creala
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Cicla attraverso tutti i file nella cartella di input
for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_directory, filename)
        output_csv_path = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}.csv')

        with (open(input_file_path, 'r') as input_file, open(output_csv_path, 'w', newline='') as output_csv):
            # Leggi il file di testo
            lines = input_file.readlines()

            # Scrivi nel file CSV con il punto e virgola come delimitatore
            csv_writer = csv.writer(output_csv, delimiter=';')
            for line in lines:
                # Qui sto suddividendo la riga in base al tab e scrivendo nel CSV
                if os.path.basename(input_file.name) == 'data_clinical_patient.txt' or os.path.basename(input_file.name) == 'data_mutations.txt':
                    csv_writer.writerow(line.strip().split(','))
                else:
                    csv_writer.writerow(line.strip().split('\t'))

        # Elimina il file di testo dopo la conversione
        # os.remove(input_file_path)

print('Trasformazione completata. File CSV creati nella cartella di output.')
