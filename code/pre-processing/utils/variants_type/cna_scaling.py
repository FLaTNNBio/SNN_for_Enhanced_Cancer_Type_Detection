import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def cna_scaling(df_path):
    df = pd.read_csv(df_path, delimiter=";", low_memory=False)

    cna_columns = [col for col in df.columns if col.endswith('_CNA')]

    df[cna_columns] = (df[cna_columns] + 2) /4

    df.to_csv("/home/musimathicslab/Detection-signature-cancer/Normals/output_variants_cna_trasposto_normalizzato_deviazione.csv", index=False, sep='\t')
    print(df[cna_columns])


if __name__ == '__main__':
    dataset = "/home/musimathicslab/Detection-signature-cancer/Normals/output_variants_trasposto_normalizzato_deviazione.csv"

    cna_scaling(dataset)
