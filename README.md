# HIM
Heterogeneous Influence Maximization in User Recommendation

## Datests

The Twitter dataset is avaiable in datasets/, which includes the graph network and the spread trajectory.

For the deepwalk embedding used to trained the ctr model in our work can be reference at [DeepInf Repository](https://github.com/xptree/DeepInf?tab=readme-ov-file)

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

## RR-IM

Since getting the probability predicted, first normalized the probability based on the details in Section 5.2. In the data provided, the normalization factor is equal to 100.

The data provided to implement the RR-IM is availiable at link [Baidu disk](https://pan.baidu.com/s/1CUMfvGCNqU3CseP7N1ax_g?pwd=qwrw). (P.S. Data provided has been normalized)

After downloading the data in the root directory, run the following code first, for the data preprocessing:

```
python utils.py
```

Then Running the following code for the RR-IM:

```
python RR_IM.py
```

## SpreadRec

Based on the same spread probability, we can aggreagte the spread influence of each node. For generating SpreadRec scores, running the code as follows:

```
python SpreadRec.py
```



