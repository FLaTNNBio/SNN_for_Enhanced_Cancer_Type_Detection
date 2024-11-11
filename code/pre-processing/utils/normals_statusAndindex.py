import pandas as pd

def take_status():
    df = pd.read_csv("/home/musimathicslab/Detection-signature-cancer/Normals2/data_clinical_patient.csv",
                     delimiter='\t', low_memory= False)
    df = df.rename(columns={'Person Neoplasm Status': 'TUMOR_STATUS'})
    df = df['TUMOR_STATUS']
    df = df[4:]
    df= df.replace({'TUMOR FREE': 0, 'WITH TUMOR': 1})

    df.to_csv("/home/musimathicslab/Detection-signature-cancer/Normals2/normals_status.csv", sep='\t', index=False)

def take_index():
    df = pd.read_csv("/home/musimathicslab/Detection-signature-cancer/Normals2/data_clinical_patient.csv",
                     delimiter='\t', low_memory=False)

    df = df.rename(columns={'Patient Identifier':'index'})
    df = df['index']
    df = df[4:]
    df.to_csv("/home/musimathicslab/Detection-signature-cancer/Normals2/normals_index.csv", sep='\t', index=False)

if __name__ == '__main__':
    take_status()
    take_index()