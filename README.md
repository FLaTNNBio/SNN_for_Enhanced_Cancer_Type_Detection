# Leveraging gene expression and genomic varation for cancer prediction using one-shot learning
The Cancer Genome Atlas (TCGA), a cancer genomics reference program, has molecularly characterized more than 20,000 primary cancer samples and paired normal samples covering 33 types of cancer.
This joint effort between the NCI and the National Human Genome Research Institute began in 2006. In the twelve years since, TCGA has generated more than 2.5 petabytes of genomic, epigenomic, transcriptomic, and proteomics. These data have led to improvements in the ability to diagnose, treat and prevent cancer by helping to establish the importance of cancer genomics.

## Contribution of this work
During the experimental process, the size of the dataset used was significantly increased in order to improve the diversity and representativeness of the data. This adjustment allowed the model to learn from a wider variety of examples, improving its generalization. In addition, adjustments were made to both models involved in this study, both the classification model and the Siamese-type model. A key element of the optimization was the implementation of the use of custom weights. This strategy allowed different weights to be assigned to different instances of the dataset based on the amount of samples present. Finally, a specification was introduced regarding the types of mutations, this allowed for greater precision in the analysis of genetic information. Numerous studies aimed at identifying a distinctive genomic signature for different types of cancer are being conducted in the current research landscape.

## Requirements (tested)

<code>pip install -r requirements.txt</code>

To install tensorflow follow this guide: <a href="https://blog.tensorflow.org/2023/11/whats-new-in-tensorflow-2-15.html">link</a><br>
To install and set up cuda and cudnn follow this guide:
<ul>
  <li><a href="https://medium.com/@yulin_li/how-to-update-cuda-and-cudnn-on-ubuntu-18-04-4bfb762cf0b8">install cuda - cudnn on Ubuntu</a></li>
  <li><a href="https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local">Cuda 12.2</a></li>
  <li><a href="https://developer.nvidia.com/rdp/cudnn-archive">cuDNN</a></li>
</ul>

## Technical informations - main.py
To run the project run the <code>main.py</code> script.

Within the <code>main.py</code> file there is the possibility to modify several parameters that will be explained later in this section (<a href="#bv">Boolean Variables</a>).
<br><br>Also in this section we introduce technical informations and installing guides!

### Download Dataset
<ul>
  <li>Download from Google Drive all the files in the folder <code>Dataset</code>: <a href="https://drive.google.com/drive/folders/1_mIUXzWdfXwy4XapOQedXiA6G9aEutZQ">LINK</a>;</li>
  <li>Files should be downloaded within a folder with the name <code>Dataset</code>;</li>
<li>Download of the Normals Dataset to paste in <code>Normals</code> folder: <a href="https://github.com/cBioPortal/datahub/tree/master/public/cesc_tcga">LINK</a>;</li>
</ul>

## Pre-Processing
Within the project, there is the <code>pre-processing</code> folder.
Within this are the <code>.py</code> files used to do preprocessing of the datasets.<br>
The folder contains subfolders for different functions.<br>
<ul>
    <li>The folder <code>data_cleaning</code> which contains the files for cleaning and formatting the dataset, also contains the files for calculating normalization and standard deviation of the values;</li><br>
    <li>The <code>merged_data</code> folder that contains the files for merging the files that will make up the final dataset (i.e., <code>data_clinical_patient</code>, <code>data_clinical_sample</code>, <code>data_cna</code>, <code>data_methylation</code>, <code>data_mrna_seq</code>, <code>data_mutation</code>);</li><br>
    <li>The <code>utils</code> folder that contains useful functions for possible necessary changes to the dataset, such as changing the csv delimiter, deleting columns, and the like </li>
</ul>
To pre-process the dataset correctly, it is mandatory to perform at least the following:
<ul>
    <li><code>traspose.py</code></li>
    <li><code>normalize.py</code></li>
    <li><code>deviazione.py</code></li>
    <li><code>add_variantType.py</code></li>
    <li><code>cna_scaling.py</code></li>
</ul>

## Pre-Processing normals
In order to work on normals patients, pre-processing has to follow a somewhat different procedure since their dateset
contains a variety of parameters that are not used by the network and therefore negligible.
<ul>
    <li>As with standard pre-processing, the first step is to transpose the dataset with <code>traspose.py</code>.</li>
    <li>As for the second step, we save the indexes and the cancer status of each normals with <code>normals_statusAndindex.py</code> we'll use them later on.</li>
    <li>Third step we perform normalization with <code>normalize.py</code>.</li>
    <li>Fourth step, we calculate the deviation with <code>deviazione.py</code>.</li>
    <li>Fifth step, we clean the dataset from columns with information not needed by the network with <code>data_cleaning_normals.py</code>. </li>
    <li>Sixth step, we calculate the variants of the genes with <code>add_variantType.py</code>.</li>
    <li>Seventh step, normalize the variant column _cna with <code>cna_scaling.py</code>.</li>
    <li>Eighth step, add the columns of which genes were not found with <code>add_missing_variants.py</code>.</li>
</ul>

### Config Path
In this script there are some path that we are going to describe now:<br>

Dataset
<ol style="list-style-type: numbers;">
  <li><code>dataset_path</code>: the dataset that we want to use (<code>SNP_DEL_INS_CNA_mutations_and_variants</code> has two);</li>
  <li><code>encoded_path</code>: the encoded of the dataset;</li>
</ol>

Classification
<ol style="list-style-type: numbers;">  
  <li><code>model_path</code>: where the model will be saved or uploaded;</li>
  <li><code>risultati_classification</code>: results of the classification;</li>
</ol>

Siamese
<ol style="list-style-type: numbers;">  
  <li><code>siamese_path</code>: where the model of the siamese network will be saved or uploaded;</li>
  <li><code>risultati_siamese</code>: results of the siamese network;</li>
</ol>

Normals
<ol style="list-style-type: numbers;">
    <li><code>normals_path</code>: the dataset that contains people with and without the disease;</li>
</ol>
<br>

If you want to change the dataset to use either <code>0030</code> or <code>0005</code> (read the paper for the meanings) you only 
need to edit the string containing <code>0030</code> or <code>0005</code> and replace it with one of the two.

For example:
```
dataset_path = ("dataset/data_mrna/SNP_DEL_INS_CNA_mutations_and_variants/"
                    "data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0030_dataPatient_mutations_and_variants.csv")      
```

Becomes

```
dataset_path = ("dataset/data_mrna/SNP_DEL_INS_CNA_mutations_and_variants/"
                    "data_mrna_v2_seq_rsem_trasposto_normalizzato_deviazione_0005_dataPatient_mutations_and_variants.csv")      
```

<br>

Or
```
model_path = "models/0030/classification/espressione_genomica_con_varianti_2LAYER/"
```

Becomes

```
model_path = "models/0005/classification/espressione_genomica_con_varianti_2LAYER/"
```

<h3 id="bv"> Boolean Variables </h3>
Always in the <code>main.py</code> script you can set some variables:

<ul>
  <li><code>only_variant = False</code>: if you use the dataset that contains only variations in gene mutations set this on <code>True</code>;</li>
  <li><code>data_encoded = False</code>: allows to generate the encoded of the dataset (if this is the first time you run the code leave the default value) 
    <ul>
      <li><code>False</code>: encoded to be generated;</li>
      <li><code>True</code>: load an encoded;</li>
    </ul>
  </li>
  <li><code>classification = True</code>: run the classification;</li>
    <li><code>classification_normals = True</code>: run the classification on normals dataset;</li>
  <li><code>siamese_net = True</code>: run the siamese network;</li>
    <li><code>siamese_normals = True</code>: run the siamese with normals dataset;</li>
    <li><code>normals_max_epsilon= False</code> <a href="#comparison">type of comparison range for normals</a></li>
    <li><code>normals_param_epsilon = True</code> <a href="#comparison">type of copmarison range for normals</a></li>
  <li><code>siamese_variants = True</code>: if you use the dataset that contains the variations in gene mutations set this on <code>True</code>;</li>

</ul>
<br>

The Siamese Network can only be launched if it has a classification model already trained and saved. In the project the classification model has already been trained. 
If you want to use the models in this project and not start experimenting again set the parameters in this way (example for <code>0030</code> dataset):
```
only_variant = False
data_encoded = True
classification = False
classification_normals = False
siamese_net = True
siamese_normals = False
siamese_variants = True
normals_max_epsilon = False
normals_param_epsilon = False
siamese_variants = False
```

## Siamese model with normal patients
To perform the operations on the normals datasets, the <strong>pre-trained siamese network</strong> found in the <code>Detection-signature-cancer/code/models/0005/siamese/espressione_genomica_con_varianti_2LAYER/</code> folder was used.<br><br>
The <code>siamese_normals</code> with normals dataset, when <code>True</code>, create a set of tresholds of similarity for each 
<code>Cancer Type</code> and, for each <code>Normal</code>, check if the patient is inclined to contract the disease of
the same type.

The comparison is made between a normals and a parameterizable number, <code>k</code>, of patients with each type of cancer.<br>
Thus a normals will be compared with each cancer type k times to calculate their propensity to get that type of cancer.<br><br>
A normal patient is more likely to get a type of cancer if his or her similarity value is in one of the ranges that we are now going to present.
 
<h3 id="comparison">Different ranges for comparisons</h3>
To determine whether a normal patient was likely to get a certain type of cancer, we decided to rely on two distinct ranges.

#### Max Epsilon
The first range is nothing more than the range between the average threshold of a cancer type and the deviation of that threshold. Thus, if a patient falls within this range, he or she is likely to contract that type of cancer.
The flag to use this type of range is <code>normals_max_espilon</code>

#### Param Epsilon
The second range, on the other hand, again consists of the average threshold of a cancer type but this time with a parameterizable value. 
The flag to use this type of range is <code>normals_param_espilon</code>
### Results

The results of each threshold calculation are saved within the <code>threshold_nameOfCancer.txt</code> file inside the <code>Threshold</code> folder.<br>
In addition, in the root folder of the project, the <code>threshold.txt</code> file containing the values of all calculated thresholds is also generated

Each of its rows is a different type of cancer and contains <code>minimum threshold</code>, <code>maximum threshold</code>, <code>average of thresholds</code>, and <code>standard deviation of thresholds</code>, respectively. <br>
So they will be displayed like this:

| Cancer_Type | Min | Max | Mean | Std |
| --- | --- | --- |------|-----|

<br>
On the other hand, the results of comparisons with normal patients is saved within the <code>Results_Comparison</code> folder.<br>
Which in turn contains the Over_Comparison folder, where all 
normal patients who have fallen within the established range of epsilon 
are saved, and the Over_Percentage folder, where normal patients 
who have fallen within the established 
range of epsilon <b>in a percentage greater than 50%</b> are saved

<br>The files inside Over_Comparison contains on each row:<br>

| Patient identifier | Calculated similarity | Type of cancer | Threshold of cancer |
|--------------------|-----------------------| --- |------|

The files inside Over_Percentage contains on each row:<br>

| Patient identifier  | Type of cancer |
|----|----|

## SHAP-enhanced SNNs: a novel mathematical perspective

### Overview

This repository provides a framework for integrating SHAP values into a Siamese Neural Network (SNN) for cancer-type prediction. The SNN computes a **similarity score** between pairs of samples, reflecting their likelihood of belonging to the same cancer type. SHAP values are used to quantify the contribution of each feature to the similarity score, providing insights into feature importance in the context of cancer classification.

### Key Concepts

1. **Similarity Score**: Given a pair of input samples `x_i` and `x_j`, the SNN computes a similarity score `S(fv(x_i), fv(x_j)) ∈ [0, 1]`, where `fv(x)` represents the feature vector of sample `x`. This score indicates the likelihood of the samples belonging to the same cancer type.
2. **SHAP Value Integration**: SHAP values quantify the contribution of individual features to the similarity score. However, since features for a pair of samples can assume different values (`fv_i(x)` and `fv_i(y)`), two SHAP values (`φ_i(x)` and `φ_i(y)`) are computed independently for each feature.
3. **Unified SHAP Value**: To summarize the importance of a feature for a sample pair `p = (x, y)`, the unified SHAP value is defined as:  
   φ_i(p) = (|φ_i(x)| + |φ_i(y)|) / 2
   This value measures the combined contribution of the feature across both samples, capturing the overall influence on the similarity score.
4. **Global Feature Importance**: For a set of sample pairs `P`, the global SHAP importance of a feature `i` is the mean unified SHAP value across all pairs in `P`, defined as:  
  Φ_i(P) = (Σ φ_i(p) for p ∈ P) / |P|

### Cancer-Specific Feature Importance

To identify the most important features for each cancer type `c`, the following methodology is applied:
1. Extract all samples corresponding to `c`.
2. Generate pairs:
   - **Positive Pair**: Two samples from the same cancer type.
   - **Negative Pair**: A sample from `c` paired with one from a different cancer type.
3. Compute \(\Phi_i(P_c)\), the cancer-specific global feature importance, using the pairs \(P_c\).

This approach considers both the similarity and dissimilarity contributions of features, leveraging all available data. By creating both positive and negative pairs, it accounts for the feature's role in distinguishing between cancer types.

### Advantages

This technique offers the following improvements:
- **Granular Insights**: Feature importance is calculated specifically for each cancer type, as opposed to the dataset-wide approach proposed in [Mostavi et al. (2021)](https://doi.org/10.1109/mostavi2021cancersiamese).
- **Gene-Level Analysis**: The method enables identifying gene-associated feature importance (e.g., gene expression, genomic mutations) for each of the 24 cancer types described in the dataset.

### Code



## Author & Contacts 

| Name | Description |
| --- | --- |
| <p dir="auto"><strong>Rocco Zaccagnino</strong> |<p dir="auto">Email - <a href="mailto:rzaccagnino@unisa.it">rzaccagnino@unisa.it</a></p>|
| <p dir="auto"><strong>Gerardo Benevento</strong> |<p dir="auto">Email - <a href="mailto:gbenevento@unisa.it">gbenevento@unisa.it</a></p>|
| <p dir="auto"><strong>Delfina Malandrino</strong> |<p dir="auto">Email - <a href="mailto:dmalandrino@unisa.it">dmalandrino@unisa.it</a></p>|
