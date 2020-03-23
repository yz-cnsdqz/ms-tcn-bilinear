# Multistage Temporal Convolutional Network with A Bilinear Module


## Acknowledgement
Our code is developped based on the following work, the github repo of which is [here](https://github.com/yabufarha/ms-tcn).

    @inproceedings{farha2019ms,
      title={Ms-tcn: Multi-stage temporal convolutional network for action segmentation},
      author={Farha, Yazan Abu and Gall, Jurgen},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={3575--3584},
      year={2019}
    }


### Training
To train from scratch, one can run

    sh script_run_trainval.sh $dataset $pooling $dropout $epoch
    
For example, to train models on the **50Salads** dataset with **RPGaussian** method and **0.5 dropout ratio** for **50 epochs**. One should run in the terminal

    sh script_run_trainval.sh 50salads RPGaussian 0.5 50

One notes that the ```dropout``` argument only works for higher order information pooling. When using **FirstOrder** pooling, which is exactly the original paper of MS-TCN, 
```dropout``` is actually not used. For example, when train the original MS-TCN model, one can run

    sh script_run_trainval.sh 50salads FirstOrder 0.5 50
in which the dropout ratio does not apply.
 

### Evaluation
After training, the checkpoints are stored in the __models__ folder. Alternatively, one can download our checkpoints, and save them into the __models__ folder. To test, one can for example run

    sh script_run_val.sh 50salads FirstOrder 0.5 50
    

Then, one will see the evaluation for individual splits and their average, such as 

       ---------------final results -------------------
        Acc: 73.8493
        Edit: 64.8312
        F1@0.10: 71.8615
        F1@0.25: 69.2641
        F1@0.50: 57.1429
        Acc: 75.8682
        Edit: 66.8940
        F1@0.10: 71.8310
        F1@0.25: 68.5446
        F1@0.50: 58.6854
        Acc: 80.5575
        Edit: 70.5910
        F1@0.10: 79.2541
        F1@0.25: 76.9231
        F1@0.50: 66.2005
        Acc: 82.5404
        Edit: 64.6688
        F1@0.10: 70.2479
        F1@0.25: 69.0083
        F1@0.50: 63.6364
        Acc: 83.1575
        Edit: 68.6581
        F1@0.10: 77.4038
        F1@0.25: 75.4808
        F1@0.50: 67.3077
        ------- overall ----------
        Acc:79.194598
        Edit:67.128601
        F1@10:74.119663
        F1@25:71.844156
        F1@50:62.594565


## Data
Please put these folders of frame-wise features in to the __data__ folder.

[download](https://drive.google.com/drive/folders/16U-rtxgSe6udBNiJPVQppjiRgjezDu9O?usp=sharing)



## checkpoints
Please put these checkpoints folders in to the __models__ folder, for reproducing the results in our manuscript.

[download](https://drive.google.com/drive/folders/1vCu3Srj90KefPDVkY3v29pX8T9FGq26l?usp=sharing)
