import os
import shutil


def copy_files(source_folder, destination_folder, file_name):
    # Verifica che la cartella di destinazione esista
    if not os.path.exists(destination_folder):
        print(f"Errore: La cartella di destinazione {destination_folder} non esiste.")
        return

    for root, dirs, files in os.walk(source_folder):
        if file_name in files:
            source_file = os.path.join(root, file_name)
            relative_path = os.path.relpath(source_file, source_folder)
            destination_file = os.path.join(destination_folder, relative_path)

            # Verifica che il file di origine esista prima di copiare
            if os.path.exists(source_file):
                shutil.copy2(source_file, destination_file)
                print(f"Copiato {file_name} da {source_file} a {destination_file}")
            else:
                print(f"Errore: Il file di origine {source_file} non esiste. Ignorato.")


# Sostituisci con il percorso della cartella principale di origine
source_folder = "/home/alberto/Scrivania/Dataset"

# Sostituisci con il percorso della cartella principale di destinazione
destination_folder = "/home/alberto/Scrivania/Dataset (completo)"

# Sostituisci con il nome del file che stai cercando
file_name = "data_mutations.csv"

copy_files(source_folder, destination_folder, file_name)
