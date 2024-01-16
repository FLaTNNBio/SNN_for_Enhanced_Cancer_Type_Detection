import pandas as pd

data = pd.read_csv('/dataset/data_mrna/'
                   'only_variants/'
                   'data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0030_dataPatient_only_variants.csv',
                   delimiter=';')

df = pd.DataFrame(data)
print(f"Shape iniziale: {df.shape}")

df = df[[col for col in df.columns if not col.endswith(('_TNP', '_ONP', '_DNP'))]]

print(f"Nuovo shape senza le 3 variazioni: {df.shape}")

df.to_csv('/home/alberto/Documenti/GitHub/Detection-signature-cancer/code/dataset/data_mrna/'
          'SNP_DEL_INS_CNA_variants/'
          'data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0030_dataPatient_only_variants.csv',
          index=False, sep=';')
