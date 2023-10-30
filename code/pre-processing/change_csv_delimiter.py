import csv

input_file = '/home/alberto/Scrivania/Dataset/blca_tcga/data_clinical_patient.csv'  # Sostituisci 'input.csv' con il nome del tuo file di input
output_file = '/home/alberto/Scrivania/Dataset/blca_tcga/data_clinical_patient.csv'  # Sostituisci 'output.csv' con il nome che desideri per il file di output

with open(input_file, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    data = list(reader)

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerows(data)