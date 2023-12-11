import pandas as pd


# Eliminiamo le prime 9 righe di data_mrna perchè sono [Not Avaiable] per
# tutti i pazienti
def elimina_prime_nove_righe(input_file, output_file):
    # Leggi l'header del CSV utilizzando pandas
    header = pd.read_csv(input_file, delimiter=';', nrows=0, low_memory=False).columns

    # Leggi il CSV escludendo le prime 9 righe
    df = pd.read_csv(input_file, delimiter=';', skiprows=10, header=None, low_memory=False)

    # Assegna l'header originale al DataFrame
    df.columns = header

    # Scrivi il DataFrame nel file di output
    df.to_csv(output_file, index=False, sep=';')

    print("Eliminazione completata.")


def trasponi_csv(input_path, output_path):
    # Carica il file CSV in un DataFrame
    df = pd.read_csv(input_path, delimiter=';', index_col=0, low_memory=False)

    # Esegui la trasposizione del DataFrame
    df_trasposto = df.transpose()

    # Salva il DataFrame trasposto in un nuovo file CSV
    df_trasposto.reset_index().to_csv(output_path, index=False, sep=';')

    print(f"Trasposizione completata. Il file trasposto è stato salvato in: {output_file_path}")


if __name__ == "__main__":
    input_file_path = "../../Dataset/data_mrna_seq_v2_rsem.csv"
    output_file_path_clean = "../../Dataset/output_elimina_righe.csv"
    output_file_path = "../../Dataset/output_trasposto.csv"

    elimina_prime_nove_righe(input_file_path, output_file_path_clean)
    trasponi_csv(output_file_path_clean, output_file_path)
