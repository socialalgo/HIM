# HIM
Heterogeneous Influence Maximization in User Recommendation

## Enviroment

```
conda create -n HIM python==3.10
conda activate HIM
pip install numpy
pip install pandas
pip install tqdm
```

## Datests

The Twitter dataset is avaiable in datasets/, which includes the graph network and the spread trajectory.

For the deepwalk embedding used to trained the ctr model in our work can be reference at [DeepInf Repository](https://github.com/xptree/DeepInf?tab=readme-ov-file)

To run the HeteroIR and HeteroIM, the data provided is availiable at link [Baidu disk](https://pan.baidu.com/s/17PbtUpIUgeOPZLyfH0Hoqg?pwd=mun5). Please download the file to the dataset/ Folder for the further execution.

## Spread Probability modeling

FuxiCTR-main-HIM directory includes the code to implement the training and predict process of the spread probability $P_{ij}$. The source repository is available at [FuxiCTR repository](https://github.com/reczoo/FuxiCTR).

```                                          
├─ data                             --  training and testing data for Precision, Recall, NDCG@k                                                                   
├─ model_zoo                        -- directory containing the AutoInt, FinalNet, EulerNet model                                     
│  ├─ EulerNet                                               
│  │  ├─ config                     -- hyperparamters                                                      
│  │  ├─ run_expid.py               -- Train file                         
│  └──── run_pred.py                -- Predict file   
├─ requirements.txt                                          
└─ setup.py                                                                 
```
For training mode, shift the dataset_id of EulerNet_default to twitter_data_train in model_config.yaml. Before running, you should unzip the file in FuxiCTR-main-HIM/data/twitter_data_train and then execute the code as follows:

```
python run_expid.py
```

For testing mode, shift the dataset_id of EulerNet_default to twitter_data_test in model_config.yaml. Before running, you should copy the extra files from the FuxiCTR-main-HIM/data/twitter_data_train to the FuxiCTR-main-HIM/data/twitter_data_test. And execute the code as follows:

```
python run_pred.py
```

## HeteroIR

Based on the probability predicted, we can aggreagte the spread influence of each node. For generating SpreadRec scores, running the code as follows:

```
python HeteroIR.py
```

## HeteroIM

To get the recommendation results obtained by HeteroIM, running the following code as follows:

```
python HeteroIM.py
```

## Evaluation

The evaluation implementation about Recall@K, NDCG@K and NSpread@K is available at evaluators.py

To get the evaluation results on EulerNet, please running the following code as follows:

```
python evaluators.py
```

## Extension

To run the HeteroIM algorithm on your own datasets, you can run the following codes for the preprocess of the network data.

Change the csv file name in preprocess.py to your file, which contains three coloums [src, tar, probability]

```
python preprocess.py
```
