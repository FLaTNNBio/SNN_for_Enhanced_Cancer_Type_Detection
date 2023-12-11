import os
import shutil

cartella_principale = "/home/alberto/Scrivania/Dataset_clean"

# Itera sui numeri da 5 a 67
for numero in range(5, 68):
    nome_file_input = f"data_mrna_v2_seq_rsem_{numero}.csv"
    percorso_file_input = os.path.join(cartella_principale, nome_file_input)

    # Controlla se il file di input esiste
    if os.path.exists(percorso_file_input):
        # Crea la sottocartella se non esiste
        cartella_destinazione = os.path.join(cartella_principale, str(numero))
        if not os.path.exists(cartella_destinazione):
            os.makedirs(cartella_destinazione)

        # Copia il file nella sottocartella corrispondente
        shutil.copy(percorso_file_input, cartella_destinazione)
        print(f"Copiato {nome_file_input} in {cartella_destinazione}")

print("Operazioni completate con successo.")
