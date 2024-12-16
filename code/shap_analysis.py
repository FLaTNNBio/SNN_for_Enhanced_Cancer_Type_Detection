import numpy as np
import pandas as pd
import random
import os
import shap
import csv
import matplotlib.pyplot as plt

import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorflow.python.keras.utils.generic_utils import custom_object_scope
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Conv1D, Flatten, MaxPooling1D
from keras.models import Sequential
from tensorflow.keras.models import Model

from keras import backend as K

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def last_layer(encoded_l, encoded_r, lyr_name='cos'):
    # Add cosine similarity function
    cos_layer = Lambda(lambda tensors: K.sum(tensors[0] * tensors[1], axis=-1, keepdims=True) /
                                       tf.math.l2_normalize(tensors[0]) * tf.math.l2_normalize(tensors[1]))

    cos_distance = cos_layer([encoded_l, encoded_r])
    tf.print(cos_distance)
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(cos_distance)

    return prediction

def get_siamese_model(input_shape, model):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    siamese_model = Sequential()

    siamese_model.add(
        Conv1D(filters=256, kernel_size=50, strides=1, activation='relu', weights=model.layers[1].get_weights(),
               padding='same', input_shape=input_shape))  # lo strides di questo era impostato a 'strides=50'
    siamese_model.add(
        Conv1D(filters=128, kernel_size=10, strides=1, activation='relu', weights=model.layers[2].get_weights(),
               padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))

    siamese_model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='sigmoid',
                             weights=model.layers[4].get_weights(), padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))

    #######################################################################################
    # Ulterior 2 layers
    siamese_model.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='sigmoid',
                             weights=model.layers[6].get_weights(), padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))

    siamese_model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='sigmoid',
                             weights=model.layers[8].get_weights(), padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))
    #######################################################################################

    siamese_model.add(Flatten())

    encoded_l = siamese_model(left_input)
    encoded_r = siamese_model(right_input)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input],
                        outputs=last_layer(encoded_l, encoded_r, lyr_name='L2'),  # prediction and cosine_similarity
                        name="siamese-network")
    return siamese_net


def initialize_bias(shape, name=None, dtype=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


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

def pre_processing(dataset_path):
    dataset_df = pd.read_csv(dataset_path, delimiter=';', low_memory=False)
    dataset_df = dataset_df.replace(',', '.', regex=True)
    dataset_df = dataset_df[dataset_df['CANCER_TYPE'] != '[Not Available]']

    # Elimina le classi di cancro che hanno meno di 9 sample
    cancers_count = dataset_df['CANCER_TYPE'].value_counts()
    valori_da_mantenere = cancers_count[cancers_count >= 9].index
    dataset_df = dataset_df[dataset_df['CANCER_TYPE'].isin(valori_da_mantenere)]

    dataset_df = dataset_df.dropna()
    dataset_df = dataset_df.reset_index(drop=True)

    category_counts = dataset_df.groupby('CANCER_TYPE').size().reset_index(name='counts')
    category_counts['percentage'] = (category_counts['counts'] / len(dataset_df) * 100).round(2)
    #print(category_counts)
    #print("\n")

    average_percentage = category_counts['percentage'].mean()
    #print(f"Media delle percentuali di campioni per classe: {average_percentage} %")

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
    '''
    dataset_df = dataset_df.drop(columns=["SEX", "SAMPLE_TYPE", "ONCOTREE_CODE", "OS_STATUS",
                                          "AJCC_PATHOLOGIC_TUMOR_STAGE", "AGE", "SOMATIC_STATUS"])
    '''
    columns_to_drop = ["SEX", "SAMPLE_TYPE", "ONCOTREE_CODE", "OS_STATUS",
                       "AJCC_PATHOLOGIC_TUMOR_STAGE", "AGE", "SOMATIC_STATUS"]
    existing_columns = [col for col in columns_to_drop if col in dataset_df.columns]
    dataset_df = dataset_df.drop(columns=existing_columns)
    ###############################################################################################

    classes = pd.unique(dataset_df["CANCER_TYPE"])
    n_classes = len(pd.unique(classes))
    del dataset_df["index"]

    #le = LabelBinarizer()

    y_df = dataset_df["CANCER_TYPE"]
    del dataset_df["CANCER_TYPE"]

    y = y_df.to_numpy()

    return dataset_df, n_classes, classes, y

def get_batch(batch_size, x_train, class_train_ind, genes_len, md='train'):
    """
    Create batch of n pairs, half same class, half different class
    """
    # Emanuele ver. : modifiche per togliere ,
    # x_train = x_train.replace(',', '.', regex=True)
    #
    # x_train = x_train.astype(np.float64)
    #
    categories = random.sample(list(class_train_ind.keys()), len(class_train_ind.keys()))
    n_classes = len(class_train_ind.keys())

    pairs = [np.zeros((batch_size, genes_len, 1)) for i in range(2)]

    targets = np.zeros((batch_size,), dtype= np.float128)

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    j = 0
    for i in range(batch_size):
        if j > n_classes - 1:
            categories = random.sample(list(class_train_ind.keys()), len(class_train_ind.keys()))
            j = 0

        category = categories[j]
        idx_1 = random.sample(class_train_ind[category], 1)[0]

        pairs[0][i, :, :] = x_train.values[idx_1].reshape(genes_len, 1).astype(np.float128)
        if i >= batch_size // 2:
            category_2 = category
            idx_2 = random.sample(class_train_ind[category_2], 1)[0]
            pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1).astype(np.float128)

        else:
            ind_pop = list(categories).index(category)
            copy_list = categories.copy()
            copy_list.pop(ind_pop)
            category_2 = random.sample(copy_list, 1)[0]
            idx_2 = random.sample(class_train_ind[category_2], 1)[0]
            pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1).astype(np.float128)

        j += 1
    return pairs, targets

def indices_save(dataset):
    # Creazione di una mappa dove la chiave è il tipo di cancro e
    # il valore è una lista degli indici delle righe
    cancer_map = {}
    for index, cancer_type in dataset['CANCER_TYPE'].items():
        if cancer_type in cancer_map:
            cancer_map[cancer_type].append(index)
        else:
            cancer_map[cancer_type] = [index]
    return cancer_map


def get_batch_for_class(cancer_type, x_train, class_train_ind, genes_len, size=2):
    """
    Dato un tipo di tumore specifico, crea un batch dove tutti i samples di quel tumore
    sono utilizzati almeno in due coppie, una positiva ed una negativa
    """

    batch_size=len(class_train_ind[cancer_type])*size

    pairs = [np.zeros((batch_size, genes_len, 1)) for i in range(2)]
    targets = np.zeros((batch_size,))
    targets[batch_size // 2:] = 1

    categories = list(class_train_ind.keys()).copy()
    categories.remove(cancer_type)
    #class_pairs=[]
    idxs_1=[]

    for i in range(0,batch_size):

        idx_1 = random.sample(class_train_ind[cancer_type], 1)[0]
        idx_1 = class_train_ind[cancer_type][i%len(class_train_ind[cancer_type])] #random.sample(class_train_ind[cancer_type], 1)[0]
        idxs_1.append(idx_1)
        pairs[0][i, :, :] = x_train.values[idx_1].reshape(genes_len, 1)

        #category_2 = cancer_type
        idx_2 = random.sample(class_train_ind[cancer_type], 1)[0]
        pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1)
        #class_pairs.append([inverse_ind[idx_1],inverse_ind[idx_2]])

    return pairs, targets

def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss



def plot_stacked_bar_chart(name,title,sorted_shap_val_features_by_gene,stacked_bar_chart_dict,rank=30):
    genes_to_plot =  {key: stacked_bar_chart_dict[key] for key in sorted_shap_val_features_by_gene[0:rank]}
    df =  pd.DataFrame.from_dict(genes_to_plot,orient='index')
    df.plot(kind="barh", stacked=True)
    plt.title(title)
    plt.savefig(name, bbox_inches='tight')

def plot_shap(name,title,shap_val,sorted_feature_names,rank=30):
    shap_to_plot=shap_val[0:rank]
    labels=sorted_feature_names[0:rank]
    fig, ax = plt.subplots()
    y_pos = np.arange(len(shap_to_plot))
    ax.barh(y_pos, shap_to_plot)
    ax.set_yticks(y_pos, labels=labels)
    plt.title(title)
    plt.savefig(name, bbox_inches='tight')


#GPU
def get_global_shap(shap_val, feature_names, by_gene=None):
  avg_shaps=[]

  # generate a list of shaps for each feature of each pair by averaging the absolute values
  for x,y in zip(shap_val[0][0],shap_val[0][1]):
    avg_sample=[]
    for i in range(0,len(shap_val[0][0][0])):
      avg = (np.abs(x[i])+np.abs(y[i]))/2
      avg_sample.append(avg)
    avg_shaps.append(avg_sample)
  avg_shaps=np.array(avg_shaps) # we have a list of averaged shaps for each feature of each input pair

  # Get a uniq, average shap for each feature
  global_shaps=[]
  for c in range(0,len(avg_shaps[0])): # for each feature
    mean_shap_f=np.mean(avg_shaps[:,c]) # average through the column
    global_shaps.append(mean_shap_f)

  if by_gene:
    genes={}
    for feature in feature_names:
      f = feature.split('_')[0]
      if f in genes.keys():
        genes[f]+=1
      else:
        genes[f]=1
    #print(genes)
    genes_shap=[]
    i=0
    for k in genes.keys():
      shap_gene=0
      count=0
      while count < genes[k]:
        shap_gene+=global_shaps[i]
        i+=1
        count+=1
        #print(i)
        #print(count)
      genes_shap.append(shap_gene)

    sorted_global_shaps, sorted_feature_names = (list(t) for t in zip(*sorted(zip(genes_shap, genes.keys()),reverse=True)))

  else:
    sorted_global_shaps, sorted_feature_names = (list(t) for t in zip(*sorted(zip(global_shaps, feature_names),reverse=True)))

  return(sorted_global_shaps, sorted_feature_names,avg_shaps,global_shaps)


def get_stacked_bar_chart_dict(shap_val,features):
  categories=['EXP','SNP', 'INS', 'DEL', 'CNA']
  genes=[]
  for feature in features:
    f = feature.split('_')[0]
    if f in genes:
      continue
    genes.append(f)

  stacked_bar_chart_dict={}
  for gene in genes:
    stacked_bar_chart_dict[gene]={}
    for cat in categories:
      stacked_bar_chart_dict[gene][cat]=0

  for shap,feature in zip(shap_val,features):
    f=feature.split('_')
    if len(f) == 1:
      stacked_bar_chart_dict[f[0]]['EXP']=shap
    else:
      stacked_bar_chart_dict[f[0]][f[1]]=shap

  return(stacked_bar_chart_dict)





def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disabilita la GPU per TensorFlow

    batch_size = 10  # Assicurati che la RAM del sistema possa gestirlo

    # Carica il modello TensorFlow
    with custom_object_scope({'initialize_bias': initialize_bias}):
        tf_siamese_model = load_model(
            '/home/musimathicslab/Detection-signature-cancer/code/models/0005/siamese/espressione_genomica_con_varianti_2LAYER/siamese_model_all_cancers.keras',
            safe_mode=False,
            custom_objects={'initialize_bias': initialize_bias}
        )

    # Carica il dataset
    dataset_df, n_classes, classes, y = pre_processing(
        '/home/musimathicslab/Detection-signature-cancer/Dataset/data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005_dataPatient_mutations_and_variants2.csv'
    )
    genes_len = dataset_df.shape[1]
    cancer_type = pd.DataFrame(y, columns=['CANCER_TYPE'])
    dataset_genes = pd.concat([dataset_df, cancer_type], axis=1)
    class_ind = indices_save(dataset_genes)
    pairs, target = get_batch(batch_size, dataset_df, class_ind, genes_len)

    input_shape = (genes_len, 1)  # Adatta la forma degli input
    print("Input shape:", input_shape)


    # Previsioni con TensorFlow
    output_tf = tf_siamese_model.predict([pairs[0], pairs[1]])
    print("Predizioni TensorFlow:", output_tf)


    explainer = shap.GradientExplainer(tf_siamese_model, pairs)


    # Calcola le spiegazioni
    shap_values = explainer.shap_values(pairs)

    print(shap_values)

    out_path = '/home/musimathicslab/Detection-signature-cancer/'
    df_features = dataset_df.columns

    for cancer in [classes]:
        # Ottieni i dati per il test
        pairs_test, target_test = get_batch_for_class(cancer, dataset_df, class_ind, genes_len, size=2)


        # Converte i dati in tensori PyTorch
        input1 = torch.tensor(pairs_test[0], dtype=torch.float32)
        input2 = torch.tensor(pairs_test[1], dtype=torch.float32)

        input1 = input1.permute(0, 2, 1)  # Da (batch_size, sequence_length, channels) a (batch_size, channels, sequence_length)
        input2 = input2.permute(0, 2, 1)

        # Combina input1 e input2
        pairs_test_torch = torch.stack((input1, input2), dim=1)
        predictions = tf_siamese_model(pairs_test_torch)
        print(predictions)

        # Calcola gli SHAP values
        shap_val = explainer.shap_values(pairs_test_torch)

        sorted_global_shaps, sorted_feature_names, _, unsorted_global_shaps = get_global_shap(shap_val, df_features)

        # Genera il plot per le feature
        plot_shap(out_path + cancer + '-shap-features.png',
                  cancer + ' - SHAP by feature',
                  sorted_global_shaps, sorted_feature_names, 20)

        # Calcola gli SHAP per ogni gene (assicurati che la funzione supporti il nuovo modello)
        gene_sorted_global_shaps, gene_sorted_feature_names, _, _ = get_global_shap(shap_val, df_features, by_gene=True)


        # Genera il plot per i geni
        plot_shap(out_path + cancer + '-shap-features-by-gene.png',
                  cancer + ' - SHAP by gene',
                  gene_sorted_global_shaps, gene_sorted_feature_names, 20)

        # Genera il grafico a barre
        stacked_bar_chart_dict = get_stacked_bar_chart_dict(unsorted_global_shaps, df_features)


        plot_stacked_bar_chart(out_path + cancer + 'stacked-barchart',
                               cancer + ' - SHAP values contributions',
                               gene_sorted_feature_names, stacked_bar_chart_dict, 20)



if __name__ == "__main__":
    main()