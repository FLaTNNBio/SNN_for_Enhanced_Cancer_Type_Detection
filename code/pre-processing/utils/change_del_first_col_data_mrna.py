import csv

input_file = '../../Dataset/data_mrna_seq_v2_rsem.csv'
output_file = '../../Dataset/out.csv'

# Apri il file di input in modalità lettura e il file di output in modalità scrittura
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    csv_reader = csv.reader(infile, delimiter=';')
    csv_writer = csv.writer(outfile, delimiter=';')

    # Leggi l'header dal file di input
    header = next(csv_reader)

    # Scrivi l'header nel file di output
    csv_writer.writerow(header)

    # Itera sulle righe del file di input
    for row in csv_reader:
        # Sostituisci il punto e virgola con la virgola nella prima colonna
        row[0] = row[0].replace(';', ',')

        # Scrivi la riga modificata nel file di output
        csv_writer.writerow(row)
