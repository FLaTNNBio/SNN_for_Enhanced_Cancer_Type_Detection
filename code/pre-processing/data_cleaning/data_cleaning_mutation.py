import pandas as pd
import os


def copia_colonne(input_csv, output_csv):
    # Label da cercare
    label_da_cercare = ["Hugo_Symbol", "Tumor_Sample_Barcode", "Chromosome", "Start_Position",
                        "End_Position", "Strand", "Consequence", "Variant_Classification", "Variant_Type"]

    # Leggi il CSV in un DataFrame pandas
    try:
        df = pd.read_csv(input_csv, delimiter=';', dtype='str', low_memory=False)
    except FileNotFoundError:
        print(f"Errore: Il file {input_csv} non è stato trovato.")
        return

    # Verifica la presenza delle label nel DataFrame
    labels_presenti = [label for label in label_da_cercare if label in df.columns]

    # Notifica a video se una label non è presente
    labels_mancanti = set(label_da_cercare) - set(labels_presenti)
    for label in labels_mancanti:
        print(f"Avviso: Label '{label}' non trovata nel CSV.")

    # Se ci sono label mancanti, esce senza creare il CSV di output
    if labels_mancanti:
        return

    # Seleziona solo le colonne con le label presenti
    df_selezionato = df[labels_presenti]

    df_selezionato.to_csv(output_csv, index=False, sep=';')
    print(f"Operazione completata per {input_csv}. Output salvato in {output_csv}.")


def modifica_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'data_mutations.csv':
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, 'data_mutations.csv')
                copia_colonne(input_path, output_path)


if __name__ == '__main__':
    directory_principale = "/home/alberto/Scrivania/Dataset"
    modifica_files(directory_principale)
