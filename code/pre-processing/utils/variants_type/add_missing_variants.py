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


def add_missing(dataset_df):
    data = pd.read_csv('/home/musimathicslab/Detection-signature-cancer/Normals2/output_variants_trasposto_normalizzato_deviazione.csv',
                       sep='\t', low_memory=False)

    c_m = dataset_df.columns.difference(data.columns)

    for col in c_m:
        data[col] = 0

    finali = data.columns.intersection(dataset_df.columns)
    print(data[finali])

    data[finali].to_csv('/home/musimathicslab/Detection-signature-cancer/Normals2/output_variants_trasposto_normalizzato_deviazione.csv', sep=';', index=False)

if __name__ == '__main__':
    dataset_df, n_classes, classes, y = fast_pre_processing('/home/musimathicslab/Detection-signature-cancer/Dataset/data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005_dataPatient_mutations_and_variants2.csv')

    add_missing(dataset_df)