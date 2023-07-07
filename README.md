# Cross-Species Transfer Learning: From Dog breed to Cat breed Classification using Deep Learning Techniques
In this project, we investigate the potential of Artificial Intelligence to enhance dog breed and cat breed classification using Transfer Leaning and Deep Learning techniques. This project is the principal component of the Deep Learning course curriculum and is part of the Master's program at National Center for Scientific Research "Demokritos" and the University of Piraeus. Our idea is to use AI to leverage knowledge gained from dog breed classification to improve the accuracy of cat breed classification. To achieve this, we try several ways of transfer learning, including varying the number of frozen layers and the learning rate. We also perform cross dataset evaluation by training on dogs and testing on cats, and vice versa. Finally, we analyze the learning curve to determine how much data is necessary to transfer to the target dataset. This simulation shows how important is the dataset size in a transfer learning training.

### About Dataset
In this study, we used two different datasets for our transfer learning task. The first one is the Stanford Dogs dataset which contains images of 120 breeds of dogs from around the world. Specifically, this dataset has been built using images and annotations from ImageNet. The second dataset we used is The Oxford-IIIT Pet Dataset with 29 categories, 17 breeds of dogs and 12 breeds of cats with roughly 200 images for each class. 

## Basic Code Information
The code has been developed in Kaggle & Google Colab. To run the code, the requirements.txt file must be used in order to load all the required modules/packages. 

## Repository Structure
The repository structure is simple and self-explanatory. It containts the following folders and files:

**Requirements Folder** - Contains the requirments file in .txt format.

**Presentation folder** - Contains the presentation in .pptx format.

**Report folder** - Contains the report as .pdf file.

**Data folder** - Contains a .txt file with the links to download the dataset, the extracted features and the saved models from Kaggle.

**Demo folder** - contains the following files
| Files/Folders    |  Description                         |              
|------------------|--------------------------------------|
| Demo-Menu.ipynb  | Main file for the demo code |
| binary_model.h5 | The saved Binary classification model to distinguish between the two classes, cats and dogs|
| dogs_model.h5  | The saved Dog breed classification model |
| cats_model.h5  | The saved Cat breed classification model |

**src folder** - contains the following files and folders
| Files/Folders         |  Description                         |              
|-----------------------|--------------------------------------|
| Cross_Species_Transfer_Learning.ipynb| Main file of our source code |