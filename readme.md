# MultiOM
Source code and datasets for OM2019 paper "Multi-view Embedding for Biomedical Ontology Matching "
* The structure of the model is shown in the figure, as follow:  
![aaaa](https://github.com/chunyedxx/MultiOM/blob/master/img/model_structure.png)
* A technical report is listed in the root directory of this project that shows an whole version of our method.
# Code
the model in our experiment are in the following scripts:  
* ontomap.py  
* ontomapsyn.py  
To train these model, please run；  
* TrainOntomap.py  
* TrainOntomapSyn.py  
To evalute the effective of our model, please run the scripts in align_evalute:  
* AlignOnto.py  
* AlignOntoSyn.py  
* AlignTfidf.py  
* AlignOntoOntoSyn.py  
* AlignOnOntoSynTf.py  
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
* DXX_MA2NCI  
  ../Datasets/DXX_MA2NCI/DXX_UQU  
  ../Datasets/DXX_MA2NCI/DXX_SYN  
* DXX_FMA2NCI  
  ../Datasets/DXX_FMA2NCI/DXX_UQU  
  ../Datasets/DXX_FMA2NCI/DXX_SYN  
# Running and parameters
Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.
## Training parameters
* The training parameters are same in the two datasets.  
  train times:1000  
  batchsize:10  
  learning rate:0.01  
  embedding size:50  
  negative rate:10  
  negative sampling:”unif”  
  optimize method:”SGD”
## Alignment process 
* Firstly, you need to run the script TrainOntomap.py; secondly, run the script TrainOntomapSyn.py.
* Nextly, you need to change the directory in AlignOnto.py or AlignOntoSyn.py and so on, and you can get the result you want.
## Alignment parameters
* The Alignment process as follows:  
![aaae](https://github.com/chunyedxx/MultiOM/blob/master/img/result3.png)  
* The details alignment parameters as follows:  
![aaaf](https://github.com/chunyedxx/MultiOM/blob/master/img/M%6042QD2_F_%5B5%7D%7D%5B%40%24%60IEA%7B1.png)  
# Results
* The comparison results of MA2NCI and FMA2NCI are shown in the following picture：
![aaab](https://github.com/chunyedxx/MultiOM/blob/master/img/result1.png)
* The results and the comparison results of MultiOM in datsset FMA2NCI is shown in the following picture：
![aaac](https://github.com/chunyedxx/MultiOM/blob/master/img/result2.png)
* you can evalute our resultsaccroding to run AlignEval.py directly.
# Citation
If you use this model or code, please cite it as follows:  
Weizhuo Li, Xuxiang Duan, Meng Wang, XiaoPing Zhang, and Guilin Qi. Multi-view Embedding for Biomedical Ontology Matching. In: OM 2019.
