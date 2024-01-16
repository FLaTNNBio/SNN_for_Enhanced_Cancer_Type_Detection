import pandas as pd


if __name__ == "__main__":
    input_file = '../../dataset/output_trasposto.csv'
    output_file = '../../dataset/output_trasposto_copy.csv'

    df = pd.read_csv(input_file, delimiter=';', low_memory=False)

    # Trova le colonne contenenti almeno un valore '[Not Available]'
    colonnes_con_valore_mancante = [colonna for colonna in df.columns if '[Not Available]' in df[colonna].values]

    # Rimuovi le colonne con valori '[Not Available]'
    df = df.drop(columns=colonnes_con_valore_mancante)

    # Stampa i risultati
    if colonnes_con_valore_mancante:
        df.to_csv(output_file, index=False, sep=';')

        print(f"Le seguenti colonne sono state rimosse poiché contenevano almeno un valore '[Not Available]':")
        for colonna in colonnes_con_valore_mancante:
            print(f"Colonna: {colonna}")
        print(f"Numero totale di colonne errate nel DataFrame: {len(colonnes_con_valore_mancante)}")
    else:
        print("Nessuna colonna contenente il valore '[Not Available]' è stata trovata.")

    # Stampa il numero totale di colonne nel DataFrame
    print(f"\nNumero totale di colonne errate nel DataFrame: {len(colonnes_con_valore_mancante)}")
    print(f"Numero totale di colonne nel DataFrame: {len(df.columns)}")
