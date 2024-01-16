import pandas as pd


def count_total_variants(dataset, data_mutation):
    # Conta il numero di occorrenze per ogni valore unico nella colonna 'Variant_Type' in 'data_mutations.csv'
    conteggio_varianti = data_mutation['Variant_Type'].value_counts()

    # Stampa i risultati
    print("\nConteggio delle varianti nel file 'data_mutations.csv':")
    print(conteggio_varianti)

    print("\nNumero totale di Features (Geni):", len(dataset.columns[9:]))
    print("Numero totale di Pazienti:", len(dataset.iloc[:, 8]))


def prepare_new_columns(dataset, data_mutation, cna):
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    #################################################################################################
    # Pre-processing 'data_mutations.csv'
    print(f"\nShape 'data_mutations.csv': {data_mutation.shape}")

    colonne_interesse_data_mutations = ['Hugo_Symbol'] + ['Tumor_Sample_Barcode'] + ['Variant_Type']
    data_mutation = data_mutation[colonne_interesse_data_mutations]

    data_mutation = data_mutation[data_mutation['Hugo_Symbol'].isin(dataset.columns[9:])]
    data_mutation = data_mutation[data_mutation['Tumor_Sample_Barcode'].isin(dataset['index'])]
    data_mutation = data_mutation.reset_index(drop=True)

    print(f"New Shape 'data_mutations.csv': {data_mutation.shape}")
    print(f"New Shape genes without duplication in 'data_mutations.csv': "
          f"{data_mutation.drop_duplicates(subset=['Hugo_Symbol']).shape[0]}")
    print(f"New Shape patients without duplication in 'data_mutations.csv': "
          f"{data_mutation.drop_duplicates(subset=['Tumor_Sample_Barcode']).shape[0]}")

    dataset = dataset[dataset['index'].isin(data_mutation['Tumor_Sample_Barcode'])]
    genes_data_mutation = data_mutation['Hugo_Symbol'].unique()
    dataset = dataset[dataset.columns[:9].tolist() + list(genes_data_mutation)]
    dataset = dataset.reset_index(drop=True)

    print(f"\nNew Shape 'dataset.csv': {dataset.shape}")
    #################################################################################################

    #################################################################################################
    # Pre-processing 'data_cna.csv'
    print(f"\nShape 'data_cna.csv': {cna.shape}")
    colonne_interesse_data_cna = ['Hugo_Symbol'] + list(dataset['index'])
    colonne_interesse_data_cna = [col for col in colonne_interesse_data_cna if col in cna.columns]
    cna = cna[colonne_interesse_data_cna]

    cna = cna[cna['Hugo_Symbol'].isin(dataset.columns[9:])]
    cna = cna.reset_index(drop=True)
    print(f"New Shape 'data_cna.csv': {cna.shape}")

    patients_cna = list(cna.columns[1:])
    dataset = dataset[dataset['index'].isin(patients_cna)]
    dataset = dataset.reset_index(drop=True)
    print(f"New Shape 'dataset': {dataset.shape}")
    #################################################################################################

    # Estrai i geni unici da ciascun dataset
    genes_dataset = set(dataset.columns[9:])
    genes_cna = set(cna["Hugo_Symbol"])

    # Trova l'intersezione dei geni comuni tra i tre dataset
    common_genes = genes_dataset.intersection(genes_cna)
    dataset = dataset[dataset.columns[:9].tolist() + list(common_genes)]
    print(f"\nNew Shape 'dataset' after intersection: {dataset.shape}")

    # Copia le prime 9 colonne nel nuovo dataframe
    data_final = dataset.iloc[:, :9].copy()
    colonne_da_aggiungere = []

    for colonna in dataset.columns[9:]:
        colonne_da_aggiungere.append(pd.Series(dataset[colonna].values, name=colonna, index=data_final.index))
        colonne_da_aggiungere.append(pd.Series(0, name=f'{colonna}_DEL', index=data_final.index))
        colonne_da_aggiungere.append(pd.Series(0, name=f'{colonna}_DNP', index=data_final.index))
        colonne_da_aggiungere.append(pd.Series(0, name=f'{colonna}_INS', index=data_final.index))
        colonne_da_aggiungere.append(pd.Series(0, name=f'{colonna}_ONP', index=data_final.index))
        colonne_da_aggiungere.append(pd.Series(0, name=f'{colonna}_SNP', index=data_final.index))
        colonne_da_aggiungere.append(pd.Series(0, name=f'{colonna}_TNP', index=data_final.index))
        colonne_da_aggiungere.append(pd.Series(0.0, name=f'{colonna}_CNA', index=data_final.index))

    data_final = pd.concat([data_final] + colonne_da_aggiungere, axis=1)
    print("Numero totale di Features con varianti: ", data_final.shape[1] - 9)
    return data_final, data_mutation, cna


def pre_process_cna(data_final, cna):
    # Add cna values in 'data_final'
    patients = cna.columns[1:]
    i = 0
    for row in cna.itertuples():
        gene = row.Hugo_Symbol
        index = row.Index

        for patient in patients:
            cna_val = cna.at[index, patient]
            if cna_val == '[Not Available]':
                cna_val = 0

            gene_cna = gene + '_CNA'
            index_var = data_final.loc[data_final['index'] == patient].index[0]
            data_final.at[index_var, gene_cna] = float(cna_val)

        if i % 50 == 0:
            print(i)
        i += 1

    data_final.to_csv('/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/dataset/data_mrna/'
                      'mutations_and_variants/'
                      'data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005_dataPatient_mutations_and_variants.csv',
                      index=False, sep=';')
    print("Data_cna copy END!\n")
    return data_final


def pre_process_mutations(data_final, data_mutation):
    # Add variant_type in 'data_final'
    for index, row in data_mutation.iterrows():
        gene = row['Hugo_Symbol']
        patient = row['Tumor_Sample_Barcode']
        variant_type = row['Variant_Type']

        gene_var = gene + '_' + variant_type
        index_var = data_final.loc[data_final['index'] == patient].index[0]
        count_var = data_final.at[index_var, gene_var]
        data_final.at[index_var, gene_var] = count_var + 1

    data_final.to_csv('/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/dataset/data_mrna/'
                      'mutations_and_variants/'
                       'data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005_dataPatient_mutations_and_variants.csv',
                      index=False, sep=';')
    print("Data_mutations copy END!")
    return data_final


if __name__ == '__main__':
    print("Lettura primo dataset...")
    data = pd.read_csv(
        '/dataset/data_mrna/'
        'deviazione_standard_dataPatient/data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005_dataPatient.csv',
        delimiter=';', low_memory=False)

    print("Lettura secondo dataset...")
    data_mutations = pd.read_csv('/dataset/data_mutations.csv', delimiter=';')

    print("Lettura terzo dataset...")
    data_cna = pd.read_csv('/dataset/data_cna.csv', delimiter=';', low_memory=False)

    print("FINE Lettura dataset!")

    count_total_variants(data, data_mutations)

    data_with_variants, data_mutations, data_cna = prepare_new_columns(data, data_mutations, data_cna)
    data_with_variants = pre_process_cna(data_with_variants, data_cna)
    data_with_variants = pre_process_mutations(data_with_variants, data_mutations)

    data_with_variants.to_csv('/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/dataset/data_mrna/'
                              'mutations_and_variants/'
                              'data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005_dataPatient_mutations_and_variants.csv',
                              index=False, sep=';')
