import os
import csv


def converti_in_csv(directory):
    i = 1
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.txt') and 'data_mrna_seq_v2_rsem.txt' in filename:
                input_file_path = os.path.join(root, filename)
                output_csv_path = os.path.join(root, f'{os.path.splitext(filename)[0]}_' + str(i) + '.csv')
                i += 1
                print(output_csv_path)

                with open(input_file_path, 'r') as input_file, open(output_csv_path, 'w', newline='') as output_csv:
                    lines = input_file.readlines()
                    csv_writer = csv.writer(output_csv, delimiter=';')
                    for line in lines:
                        csv_writer.writerow(line.strip().split('\t'))

                # Puoi rimuovere i file di testo dopo la conversione se desideri
                # os.remove(input_file_path)

    print('Trasformazione completata!')


if __name__ == '__main__':
    directory_principale = '/home/alberto/Scrivania/Dataset'
    converti_in_csv(directory_principale)
