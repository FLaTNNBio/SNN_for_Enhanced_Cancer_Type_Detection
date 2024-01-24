# Detection Signature Cancer
The Cancer Genome Atlas (TCGA), a cancer genomics reference program, has molecularly characterized more than 20,000 primary cancer samples and paired normal samples covering 33 types of cancer.
This joint effort between the NCI and the National Human Genome Research Institute began in 2006. In the twelve years since, TCGA has generated more than 2.5 petabytes of genomic, epigenomic, transcriptomic, and proteomics. These data have led to improvements in the ability to diagnose, treat and prevent cancer by helping to establish the importance of cancer genomics.

# Contribution of this work
During the experimental process, the size of the dataset used was significantly increased in order to improve the diversity and representativeness of the data. This adjustment allowed the model to learn from a wider variety of examples, improving its generalization. In addition, adjustments were made to both models involved in this study, both the classification model and the Siamese-type model. A key element of the optimization was the implementation of the use of custom weights. This strategy allowed different weights to be assigned to different instances of the dataset based on the amount of samples present. Finally, a specification was introduced regarding the types of mutations, this allowed for greater precision in the analysis of genetic information. Numerous studies aimed at identifying a distinctive genomic signature for different types of cancer are being conducted in the current research landscape.

## Requirements (tested)

| Module               | Version |
|----------------------|---------|
| tensorflow           | 2.15.0  |
| torch                | 2.1.2   |
| cuda                 | 12.2    |

To install tensorflow follow this guide: <a href="https://blog.tensorflow.org/2023/11/whats-new-in-tensorflow-2-15.html">link</a><br>
To install and set up cuda and cudnn follow this guide:
<ul>
  <li><a href="https://medium.com/@yulin_li/how-to-update-cuda-and-cudnn-on-ubuntu-18-04-4bfb762cf0b8">install cuda - cudnn on Ubuntu</a></li>
  <li><a href="https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local">Cuda 12.2</a></li>
  <li><a href="https://developer.nvidia.com/rdp/cudnn-archive">cuDNN</a></li>
</ul>

# Related work
For more information about our research project access the paper here: <a href="https://github.com/Alberto-00/Detection-signature-cancer/blob/main/exam/Bioinformatica_Paper_CancerDetection.pdf">our paper</a><br>
To view the other papers that have contributed to the cancer research study and on which we have commented follow this link: <a href="https://github.com/Alberto-00/Detection-signature-cancer/tree/main/papers">other papers</a><br>

# Author & Contacts 

| Name | Description |
| --- | --- |
| <p dir="auto"><strong>Alberto Montefusco</strong> |<br>Developer - <a href="https://github.com/Alberto-00">Alberto-00</a></p><p dir="auto">Email - <a href="mailto:a.montefusco28@studenti.unisa.it">a.montefusco28@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alberto-montefusco">Alberto Montefusco</a></p><p dir="auto">My WebSite - <a href="https://alberto-00.github.io/">alberto-00.github.io</a></p><br>|
| <p dir="auto"><strong>Alessandro Macaro</strong> |<br>Developer   - <a href="https://github.com/mtolkien">mtolkien</a></p><p dir="auto">Email - <a href="mailto:a.macaro@studenti.unisa.it">a.macaro@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alessandro-macaro-391b7a214/">Alessandro Macaro</a></p><br>|
