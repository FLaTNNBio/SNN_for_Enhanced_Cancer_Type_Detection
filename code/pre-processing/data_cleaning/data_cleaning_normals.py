import pandas as pd

def del_genes_constant(df):
    # Creare un set di geni di base (senza suffissi)
    base_genes = set(gene.split('_')[0] for gene in df)

    # Filtrare per mantenere solo i geni che hanno tutte e quattro le varianti
    filtered_genes = [gene for gene in base_genes if
                      all(any(gene_variant == gene + suffix for gene_variant in df)
                          for suffix in ["_DEL", "_SNP", "_CNA", "_INS"])]

    filtered_genes_complete = []
    for gene in filtered_genes:
        filtered_genes_complete.extend([gene + suffix for suffix in ["_DEL", "_SNP", "_CNA", "_INS"]])

    return filtered_genes_complete, list(filtered_genes)

def fast_pre_processing(dataset_path):
    dataset_df = pd.read_csv(dataset_path, delimiter=';', low_memory=False)
    dataset_df = dataset_df[dataset_df['CANCER_TYPE'] != '[Not Available]']

    # Elimina le classi di cancro che hanno meno di 9 sample
    cancers_count = dataset_df['CANCER_TYPE'].value_counts()
    valori_da_mantenere = cancers_count[cancers_count >= 9].index
    dataset_df = dataset_df[dataset_df['CANCER_TYPE'].isin(valori_da_mantenere)]

    dataset_df = dataset_df.dropna()
    dataset_df = dataset_df.reset_index(drop=True)

    category_counts = dataset_df.groupby('CANCER_TYPE').size().reset_index(name='counts')
    category_counts['percentage'] = (category_counts['counts'] / len(dataset_df) * 100).round(2)
    print(category_counts)
    print("\n")

    average_percentage = category_counts['percentage'].mean()
    print(f"Media delle percentuali di campioni per classe: {average_percentage} %")

    constant_columns = [col for col in dataset_df.columns if dataset_df[col].nunique() == 1]
    constant_columns_genes = [element for element in constant_columns if element != "SOMATIC_STATUS"]
    constant_columns_genes = [element for element in constant_columns_genes if element != "SAMPLE_TYPE"]

    result, genes = del_genes_constant(constant_columns_genes)
    if len(result) > 0:
        columns = result + genes
        dataset_df = dataset_df.drop(columns=columns)
    else:
        dataset_df = dataset_df.drop(columns=constant_columns_genes)

    ###############################################################################################
    dataset_df = dataset_df.drop(columns=["SEX", "SAMPLE_TYPE", "ONCOTREE_CODE", "OS_STATUS",
                                          "AJCC_PATHOLOGIC_TUMOR_STAGE", "AGE", "SOMATIC_STATUS"])
    ###############################################################################################

    classes = pd.unique(dataset_df["CANCER_TYPE"])
    n_classes = len(pd.unique(classes))
    del dataset_df["index"]

    #le = LabelBinarizer()

    y_df = dataset_df["CANCER_TYPE"]
    del dataset_df["CANCER_TYPE"]

    y = y_df.to_numpy()

    return dataset_df, n_classes, classes, y


def data_cleaning_normals(dataset_df):
    cna = pd.read_csv("/home/musimathicslab/Detection-signature-cancer/Normals2/data_cna.csv", sep="\t")
    output = pd.read_csv("/home/musimathicslab/Detection-signature-cancer/Normals2/output_trasposto_normalizzato_deviazione.csv", sep="\t", low_memory=False)

    cna = cna.drop('Entrez_Gene_Id', axis=1)
    cna.rename(columns={"Hugo_Symbol":"index"}, inplace=True)

    cna = cna.set_index('index').T.rename_axis('index').rename_axis(None, axis=1).reset_index()
    print(output.shape)

    colonne_mancanti = cna.columns.difference(output.columns)

    output = pd.concat([output, cna[colonne_mancanti]], axis=1)
    colonne_comuni = output.columns.intersection(dataset_df.columns)

    output = pd.concat([output['index'], output[colonne_comuni]], axis=1)

    output.to_csv('/home/musimathicslab/Detection-signature-cancer/Normals2/output_pre_variants.csv', sep="\t", index=False)

if __name__ == '__main__':
    dataset_df, n_classes, classes, y = fast_pre_processing(
        '/home/musimathicslab/Detection-signature-cancer/Dataset/data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005_dataPatient_mutations_and_variants2.csv')

    data_cleaning_normals(dataset_df)