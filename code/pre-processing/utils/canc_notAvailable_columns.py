import pandas as pd


def conta_not_available_Columns(output_file):
    df = pd.read_csv(output_file, delimiter=';', low_memory=False)

    # Inizializza una lista per memorizzare i riferimenti delle colonne con '[Not Avaiable]'
    colonnes_con_valore_mancante = []

    # Itera attraverso le colonne del DataFrame
    for colonna in df.columns:
        # Verifica se '[Not Avaiable]' è presente nella colonna corrente
        if '[Not Available]' in df[colonna].values:
            # Se presente, aggiungi il riferimento della colonna alla lista
            colonnes_con_valore_mancante.append(colonna)

    # Stampa i risultati
    if colonnes_con_valore_mancante:
        print(f"Il valore '[Not Avaiable]' è presente nelle seguenti colonne:")
        for colonna in colonnes_con_valore_mancante:
            print(f"Colonna: {colonna}")
    else:
        print("Nessuna colonna contiene il valore '[Not Avaiable]'.")

    # Stampa il numero totale di colonne nel DataFrame
    print(f"Numero totale di colonne errate nel DataFrame: {len(colonnes_con_valore_mancante)}")
    print(f"Numero totale di colonne nel DataFrame: {len(df.columns)}")


if __name__ == "__main__":
    input_file = '../../Dataset/output_trasposto.csv'
    output_file = '../../Dataset/output_trasposto_copy.csv'

    # Carica il dataset
    df = pd.read_csv(input_file, delimiter=';', low_memory=False)

    # Trova le colonne contenenti almeno un valore '[Not Available]'
    colonnes_con_valore_mancante = [colonna for colonna in df.columns if '[Not Available]' in df[colonna].values]

    # Rimuovi le colonne con valori '[Not Available]'
    df = df.drop(columns=colonnes_con_valore_mancante)

    # Stampa i risultati
    if colonnes_con_valore_mancante:
        # Scrivi il DataFrame nel file di output
        df.to_csv(output_file, index=False, sep=';')

        print(f"Le seguenti colonne sono state rimosse poiché contenevano almeno un valore '[Not Available]':")
        for colonna in colonnes_con_valore_mancante:
            print(f"Colonna: {colonna}")
        print(f"Numero totale di colonne errate nel DataFrame: {len(colonnes_con_valore_mancante)}")
    else:
        print("Nessuna colonna contenente il valore '[Not Available]' è stata trovata.")



