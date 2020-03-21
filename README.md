# Multistage Temporal Convolutional Net with Bilinear Residual Module


## Acknowledgement
Our code is developped based on the following work, the github repo of which is [here](https://github.com/yabufarha/ms-tcn).

    @inproceedings{farha2019ms,
      title={Ms-tcn: Multi-stage temporal convolutional network for action segmentation},
      author={Farha, Yazan Abu and Gall, Jurgen},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={3575--3584},
      year={2019}
    }


## Implementation
Our implementation is based on Python 3 and PyTorch 1.1 in Ubuntu 18.04. One perhaps needs to modify the **model_dir** in the **main.py** file. 

### Training
To train from scratch, one can run

    sh script_run_trainval.sh $dataset $pooling $dropout $epoch
    
For example, to train on the **50Salads** dataset with **RPGaussian** method and **0.7 dropout ratio** for **70 epochs**. One should run in the terminal

    sh script_run_trainval.sh 50salads RPGaussian 0.7 70
 
 
### Testing
To reproduce Tab.2 in our manuscript, one needs to download our checkpoints, and save them into the __models__ folder. Then, one can perform inference by for example running

    sh script_run_val.sh 50salads RPGaussian 0.7 70
    

Then, one will see the evaluation for individual splits, such as 

        ---------------final results -------------------
        Acc: 77.2142
        Edit: 64.9321
        F1@0.10: 72.3493
        F1@0.25: 69.8545
        F1@0.50: 59.4595
        Acc: 80.5537
        Edit: 74.8293
        F1@0.10: 80.8717
        F1@0.25: 78.4504
        F1@0.50: 68.7651
        Acc: 80.9786
        Edit: 70.8362
        F1@0.10: 78.4689
        F1@0.25: 77.0335
        F1@0.50: 69.3780
        Acc: 80.9601
        Edit: 71.7944
        F1@0.10: 78.3599
        F1@0.25: 75.6264
        F1@0.50: 66.9704
        Acc: 83.4894
        Edit: 72.5606
        F1@0.10: 81.7043
        F1@0.25: 78.1955
        F1@0.50: 68.6717


The final performance is given by averaging each metric score.



## Data
Please put these two folders in to the __data__ folder.

[download](https://drive.google.com/drive/folders/16U-rtxgSe6udBNiJPVQppjiRgjezDu9O?usp=sharing)



## checkpoints
Please put these checkpoints folders in to the __models__ folder, for reproducing the results in Tab. 2.

[download](https://drive.google.com/drive/folders/1vCu3Srj90KefPDVkY3v29pX8T9FGq26l?usp=sharing)
