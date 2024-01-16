import os
import csv


def cambia_label_header(path_cartella_principale, new_label):
    # Trova tutti i file 'data_methylation.csv' nelle sottocartelle
    for cartella, sottocartelle, file in os.walk(path_cartella_principale):
        for nome_file in file:
            if nome_file == 'data_methylation_hm450.csv':
                # Costruisci il percorso completo del file
                percorso_file = os.path.join(cartella, nome_file)

                # Leggi il file CSV e modifica la label dell'header della prima colonna
                with open(percorso_file, 'r', newline='') as file_csv:
                    reader = csv.reader(file_csv, delimiter=';')
                    righe = list(reader)

                    if len(righe) > 0 and len(righe[0]) > 0:
                        if righe[0][0] != new_label:
                            # Modifica la label dell'header della prima colonna
                            righe[0][0] = new_label

                            # Scrivi le modifiche nel file CSV
                            with open(percorso_file, 'w', newline='') as file_csv_modificato:
                                writer = csv.writer(file_csv_modificato, delimiter=';')
                                writer.writerows(righe)

                            print(f"Label cambiata nel file: {percorso_file}")


if __name__ == "__main__":
    cartella_principale = '/home/alberto/Scrivania/dataset (completo)'
    nuova_label = 'Hugo_Symbol'
    cambia_label_header(cartella_principale, nuova_label)
