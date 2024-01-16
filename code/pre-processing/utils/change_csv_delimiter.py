import csv

input_file = '/home/alberto/Scrivania/dataset/blca_tcga/data_clinical_patient.csv'
output_file = '/home/alberto/Scrivania/dataset/blca_tcga/data_clinical_patient.csv'

with open(input_file, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    data = list(reader)

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerows(data)
