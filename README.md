# <p align=center>`CoLLT: Contrastive Learning for Long-document Transformers`</p>
`CoLLT` is a contrastive learning framework for training BERT and its variants to classify long input sequence.

## How to run the file
1. Download the zip folder for the code
2. Run the main.py notebook to execute the experiments

## About the code
The code is written in a well designed modular way so that implementing new contrastive loss, augmentation technique or data encoder is easy. The code is divided into 4 main files: 
1. augmenters.py: contains code for different augmentation techniques which are needed for view construction
2. models.py: contains code for different data encoders
3. contrast\_models.py: contains pre processing step (like sample positive and negative samples, etc) before applying contrastive loss
4. losses.py: contains code for barlow twin's loss. We have a main.py file which handles the training of end-to-end model pipeline.

### Other files:
5. Bert_baseline.ipynb: contains code to run the BERT baseline model
6. Data_filter.ipynb: contains code for data preprocessing
7. baselines.py: conatines baseline models
8. baselines.ipynb: used to run baseline.py
9. data.pickle, data_val.pickle, data_test.pickle: contains train, validation and test data
10. data_visualization.ipynb: contains data visualization tools and techniques
