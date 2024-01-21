import pandas as pd


def drop_genes_mutations(data):
    cols_to_drop = [i for i in range(9, len(data.columns), 8)]
    data.drop(data.columns[cols_to_drop], axis=1, inplace=True)

    return data


if __name__ == "__main__":
    print("Lettura dataset...")
    dataset = pd.read_csv('/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/dataset/'
                          'data_mrna/mutations_and_variants/'
                          'data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0030_dataPatient_mutations_and_variants.csv',
                          delimiter=';', low_memory=False)

    print("FINE Lettura dataset!")

    data_final = drop_genes_mutations(dataset)
    data_final.to_csv('/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/dataset/data_mrna/only_variants/'
                      'data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0030_dataPatient_only_variants.csv',
                      index=False, sep=';')
