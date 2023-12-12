import pandas as pd


def merge_csv(input_csv1, input_csv2, output_csv):
    # Carica i due CSV in due DataFrame distinti
    df1 = pd.read_csv(input_csv1, delimiter=';')
    df2 = pd.read_csv(input_csv2, delimiter=';')

    # Esegui la fusione basata sulla colonna 'index' e 'SAMPLE_ID'
    merged_df = pd.merge(df1, df2, left_on='SAMPLE_ID', right_on='index', how='inner')

    # Elimina la colonna duplicata 'SAMPLE_ID'
    merged_df.drop('SAMPLE_ID', axis=1, inplace=True)
    print(merged_df)

    # Salva il DataFrame risultante in un nuovo CSV
    merged_df.to_csv(output_csv, index=False, sep=';')


# Esempio di utilizzo
merge_csv('/home/musimathicslab/Desktop/Cancer/Dataset/data_clinical_patient.csv',
          '/home/musimathicslab/Desktop/Cancer/Dataset/data_mrna/data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005.csv',
          '/home/musimathicslab/Desktop/Cancer/Dataset/data_mrnadata_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005_dataPatient.csv')

