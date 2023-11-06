import csv


def riempi_campi_vuoti(file_input, file_output):
    with open(file_input, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        header = next(reader)  # Ignora l'header

        with open(file_output, 'w', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=';')
            writer.writerow(header)  # Scrivi l'header nel nuovo file

            for row in reader:
                for i in range(len(row)):
                    if not row[i]:
                        row[i] = '[Not Available]'
                        print(row)

                writer.writerow(row)


# Usa la funzione con il tuo file CSV di input e di output
riempi_campi_vuoti('/home/musimathicslab/Desktop/Dataset/data_mrna_seq_v2_rsem.csv',
                   '/home/musimathicslab/Desktop/Dataset/data_mrna_seq_v2_rsemm.csv')
