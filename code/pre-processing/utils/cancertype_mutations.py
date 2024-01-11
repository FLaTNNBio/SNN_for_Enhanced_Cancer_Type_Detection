import pandas as pd


def aggiungi_colonna_cancer_type(file1, file2, output_file):
    df1 = pd.read_csv(file1, delimiter=';')
    df2 = pd.read_csv(file2, delimiter=';')

    # Unisci i dataframe in base alla colonna 'Tumor_Sample_Barcode'
    df_merged = pd.merge(df1, df2, left_on='Tumor_Sample_Barcode', right_on='SAMPLE_ID', how='left')

    # Aggiungi la colonna 'Cancer_Type' al dataframe originale
    df1['CANCER_TYPE'] = df_merged['CANCER_TYPE']

    df1.to_csv(output_file, index=False, sep=';')


aggiungi_colonna_cancer_type('/home/musimathicslab/Desktop/Cancer/Dataset/data_mutations.csv',
                             '/home/musimathicslab/Desktop/Cancer/Dataset/data_clinical_patient.csv',
                             '/home/musimathicslab/Desktop/output.csv')
