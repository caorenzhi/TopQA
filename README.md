# TopQA

TopQA is a novel method aimed to tackle a crucial step in the protein prediction problem - assessing the quality of generated protein structures. It analyzes the topology of the predicted structure, and process this information based on Convolutional Neural Network (CNN) to predict the quality of this structure.

# Citation
--------------------------------------------------------------------------------------
Smith, John, Matthew Conover, Natalie Stephenson, Jesse Eickholt, Dong Si, Miao Sun, and Renzhi Cao. "TopQA: a topological representation for single-model protein quality assessment with machine learning." International Journal of Computational Biology and Drug Design 13, no. 1 (2020): 144-153.

# Test Environment
--------------------------------------------------------------------------------------
Ubuntu, Centos

# Requirements
--------------------------------------------------------------------------------------
(1). Python3.5

(2). TensorFlow 
```
sudo pip install tensorflow
```
GPU is NOT needed.

(3) Install Keras:
```
sudo pip install keras
```

(4) Install the h5py library:  
```
sudo pip install python-h5py
```

# Run software
--------------------------------------------------------------------------------------
You could provide one PDB format model or a folder with several PDB format models for this software. Here are examples to test:

#cd script

#python3 TopQA_pred.py ../resources/pdb2img ../resources/Model_520.01 ../test/T0859_stage1_true/T0859.pdb ../test/Prediction_singleModel

#python3 TopQA_pred.py ../resources/pdb2img ../resources/Model_520.01 ../test/T0859_stage1_true/stage1 ../test/Prediction_ModelPool

You should be able to find the output file named TopQAScores.txt in the output folder.


--------------------------------------------------------------------------------------
Developed by John Smith and Dr. Renzhi Cao at Pacific Lutheran University:

Please contact Dr. Cao for any questions: caora@plu.edu (PI)
