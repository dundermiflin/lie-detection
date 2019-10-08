# lie-detection
A Deep Learning model for fact-checking comments by politicians

## Table Of Contents
-  [Overview](#overview)
-  [Details](#details)
-  [Results](#results)
-  [Future Work](#future-work)
-  [Acknowledgments](#acknowledgments)

## Overview
This is a Keras/Tensorflow implementation of one of the methods discussed in the paper [Where is your Evidence: Improving Fact-checking by Justification
Modeling](http://www.cs.columbia.edu/~sdp2137/papers/evidence_paper.pdf)

The dataset we use is the LIARPLUS which has sentences extracted automatically from the full-text verdict report written by journalists in Politifact. You can find more details about it [here](https://github.com/Tariq60/LIAR-PLUS).

## Details
We implement the *S+M* method specified in the paper to classify sentences.
There are two classification tasks - binary and multiclass.  
The dataset has the following labels:
  1. true
  2. half-true
  3. mostly-true
  4. barely-true
  5. false
  6. pants-fire

For the binary classification task, we split/map the classes as follows:
The dataset has the following labels:
  * true:
    1. true
    2. half-true
    3. mostly-true
  * false:
    1. barely-true
    2. false
    3. pants-fire
    
### Model Architecture
We use Statement Representation(_S_) and(_+_) Metadata(_M_) features as the inputs to our deep learning model.  
The metadata features of our interest are the total credit history count, including the current statement:  
1. barely true counts.
2. false counts.
3. half true counts.
4. mostly true counts.
5. pants on fire counts.  

We clean the text before-hand to remove punctuation and other unwanted characters.
There is an embedding layer to generate a 200-dimensional real-valued vector representation of the sentence. This layer is not frozen and will learn the embeddings during training.  
This is then followed by a Bi-LSTM layer to learn contextual features in both directions.  
We concatenate the sentence-level features with the metadata features, and run them through a couple of fully-connected layers.  

Below is the complete model architecture:  
![Model Architecture](https://raw.githubusercontent.com/dundermiflin/lie-detection/figures/nn_model_plot.png)

