# Few-shot learning - An application of Convolutional Neural Networks  
In this github repo we present our project implementation for the course Advanced Machine Learning @ENSAE, lecturer Austin Stromme.

## General idea
The project aims to explore the integration of Convolutional Neural Networks (CNNs) with Few-Shot Learning techniques in the context of computer vision. The focus is on addressing challenges such as image classification and object recognition with limited labeled data. The project makes use of the concept of Siamese Neural Networks. The implementation utilizes the miniImageNet dataset, preprocesses the data, and employs an extended Siamese model to achieve its goals.

## Requirements
To install the used python libraries, please use the requirements.txt file

``` 
$ pip install -r requirements.txt
```

## Data
The `few-shot_learning` folder contains all the code (in a separated sub-folder) as well as the requirements necessary for the execution (see above). Within the `code` folder, you can first of all find the folder `data_10classes` that contains 10 classes of pictures from the miniImageNet dataset that is utilized for this project. These pictures are the input for the preprocessing. After preprocessing, the `preprocessed_data6` folder will appear and contain the output of the preprocessing which is also the input for the training. For more details, please check out the report available in the root directory besides this `README.md`. Last but not least, the code directory contains two notebooks - `preprocessing.ipynb` and `training.ipynb`. Please see below to find out how to use them.

## Preprocess data
To reproduce the result of the project team, please focus on the `preprocessing.ipynb` notebook first. In case you don't have the required packages installed, feel free to do so using the command above before running the first cell. If you have installed all the requirements, feel free to run all the cells of the preprocessing file from top to bottom. If everything went well, you should be able to find the folder `preprocessed_data6` with different sub-folders for support and query pictures of each class. No further action is required in the preprocessing notebook.

## Model 
Next, please switch to the `training.ipynb` notebook. Here, please run all the cells until the one that defines the train()-function, including this one too. The next cell should be the one that contains train(train_data_iter, EPOCHS) and siamese_model.save(file_name). Here, you have the choice between two options. Either you can train your own model or you choose to use a model that is already trained. Please have a look at the two sections below.

### Train new model
If you don't have any model yet and / or you want to train a new model, please make sure that the code for the model training is not commented out. Feel free to adjust the name of your model here. Then run this cell. In the following cell, please make sure that you have the right name included when trying to load a model. Now, run all the following cells.

### Import trained model
In case you don't want to train a new model and prefer to import one instead, please include it in the folder `trained_models_105`. Besides that, make sure that you have the train() function commented out. Please have a look at the comments of this code cell as well. Make sure to specify the correct name in this case as well. Then, run all the cells below.

For further details on how our model is build, how it is evaluated etc., have a look at the markdowns, comments and outputs of the notebook or go to our report directly. 

### Comments on how the code came to life
Our data preprocessing was made from scratch. 
For the implementation of our CNN we beased our code on the code snippets provided on
[this website](https://medium.com/@prabhattgs12345789/siamese-neural-network-enhancing-ai-capabilities-with-pairwise-comparisons-4f00e2dd8256). 
As this is the implementation of a siamese network we did put quite some work into transforming the binary classification into a classification for multiple classes. For example a new interpretation of the labeling of our dataset was necessary, which led us to introduce support and query sets. Also the distance layer which measures the distance between the embeddings needed a new interpretation. 
Also the validation needed some new considerations as we created an unbalanced dataset. 
