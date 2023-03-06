# VDE

> (ISSTA'23) Interpreters for GNN-based Vulnerability Detection: Are We There Yet?

## Dataset

The Dataset we used in the paper:
Fan et al / MSR'20: https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing

## Requirement

Please check all requirements in the requirement.txt

## Preprocess
1.Run preprocess file folder  ```raw_data_preprocess.py``` to get codes from MSR dataset.

2.Run preprocess/code_normalize file folder ```normalization.py``` to normalize the codes.

3.Use joern to generate PDG graphs, we use v1.1.172, please go to Joern's website: https://github.com/joernio/joern for more details on graph generation.

  We give py scripts in preprocess file folder ```joern_graph_gen.py```.You can refer to the required file.(.bin/.dot/.json)
  
4.Run preprocess file folder ```train_w2v.py``` to get trained w2v model.

5.Run preprocess file folder ```joern_to_devign.py``` to get the data required by the VD model


## Training of vulnerability detection model
1. All codes in ```vul_detect ``` file folder.
 
2. You need to modify ```data_loader/dataset.py ```.Pay attention to split the training set and test set(like train_set.txt/test.txt to provide data path)
 
3. You need to modify ```main.py ``` and ```trainer.py ``` like some input or output paths.
 
4. Run ```main.py ``` to train or test.

## Use interpreter to give Interpretation
1.All interpreter in ```vul_explainer ```file folder.

2.All interpreter has its readme.You can refer it. We give '##' to point out which one need to modify.

3.In DeepLIFT,GNN-LRP,GNNExplainer,GradCAM:you need to modify ```benchmark/args.py```,```benchmark/data/dataset_gen.py ```,```benchmark/models/explainers.py``` and run ```benchmark/kernel/pipeline.py```

4.In PGExplainer: you need to modify ```Configures.py```,```load_dataset.py```,```metrics.py``` and run ```pipeline.py```

5.In SubgraphX: you need to modify ```/forgraph/subgraphx.py```, ```Configures.py```,```load_dataset.py``` and run ```/forgraph/subgraphx.py```


---

