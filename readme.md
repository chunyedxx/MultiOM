# MultiOM
Source code and datasets for OM2019 paper "Multi-view Embedding for Biomedical Ontology Matching "
The structure of the model is shown in the figure, as follow:
## Model structure 
The structure of the model is shown as the figure:
![aaaa](https://github.com/chunyedxx/MultiOM/blob/master/img/model_structure.png)
# Code
the model in our experiment are in the following scripts:  
* ontomap.py  
* ontomap_syn.py  
To train these model, please run；  
* train_ontomap.py  
* train_ontomap_syn.py  
To evalute the effective of our model, please run the scripts in align_evalute:  
* align_onto.py  
* align_onto_syn.py  
* align_tfidf.py  
* align_onto_ontosyn.py  
* align_onto_ontosyn_tfidf.py  
## Dependencies
* Python 3  
* Tensorflow (>=1.2)  
* Numpy
# Datasets
In our experiments, we use the Medical Ontologys FMA, NCI,MA and SNOMED,FMA,NCI.The detail of our datasets are in the floder Datasets, you can get the whole datasets in it.
## DXX_MA2NCI
The dataset DXX_MA2NCI, which realizes the alignment MA to NCI,and uses the ontology FMA as the bridge.The files are: 
* DXX_FMA
* DXX_NCI
* DXX_MA
* DXX_SYN
* DXX_UQU
## DXX_FMA2NCI
The dataset DXX_FMA2NCI, which realizes the alignment FMA to NCI,and uses the ontology SNOMED as the bridge.The files are: 
* DXX_SNOMED
* DXX_NCI
* DXX_FMA
* DXX_SYN
* DXX_UQU
## Directory structure
the train data are in the directory as follows, you can train the different datasets accroding to change the directory.  
* ../Datasets/DXX_UQU  
* ../Datasets/DXX_SYN
# Running and parameters
Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.
## Training parameters
* The training parameters are same in the two datasets.
** train times:1000
** batchsize:10
** learning rate:0.01
** embedding size:50
** negative rate:10
** negative sampling:”unif”
** optimize method:”SGD”
## Matching process 
* Firstly, you need to run the script train_ontomap.py; secondly, run the script train_ontomapsyn.py.
* Nextly, you need to change the directory in align_onto.py or align_ontosyn.py and so on, and you can get the result you want.
## Matching parameters
## results
* The detailed results of MultiOM in datsset MA2NCI is shown in the following picture.
![aaab](https://github.com/chunyedxx/MultiOM/blob/master/img/result1.png)
* The results and the comparison results of MultiOM in datsset FMA2NCI is shown in the following picture.
![aaac](https://github.com/chunyedxx/MultiOM/blob/master/img/result2.png)
# Citation
If you use this model or code, please cite it as follows:  
Weizhuo Li, Xuxiang Duan, Meng Wang, XiaoPing Zhang, and Guilin Qi. Multi-view Embedding for Biomedical Ontology Matching. In: OM 2019.
