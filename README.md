# Transformer-based-sentiment-analysis

## Introduction

This report documents the process of building a text classification model using a Transformer architecture for sentiment analysis on the IMDb movie reviews dataset. The task is to classify reviews as either positive or negative based on the sentiment expressed in the text.

## Dataset

The IMDb movie reviews dataset is used for this project. The dataset is available through TensorFlow Datasets (`tfds`). It consists of movie reviews labeled with binary sentiment (positive or negative).

## Data Preprocessing

1. **Loading Data:** The IMDb dataset is loaded using TensorFlow Datasets (`tfds`). The training and testing sets are extracted.

2. **Text Cleaning and Preprocessing:** The text data is cleaned and preprocessed using the following steps:
   - Convert text to lowercase.
   - Remove special characters, numbers, and punctuation using regular expressions.
   - Remove English stop words using NLTK.

## Model Architecture

The model is built using a Transformer architecture, which has shown great success in natural language processing tasks. The architecture includes the following components:

- **Token and Position Embedding Layer:** Embeds tokens and positional information.
- **Transformer Block:** A self-attention mechanism with feed-forward layers and layer normalization.

The model architecture is compiled with the Adam optimizer and binary cross-entropy loss. The training process includes early stopping to prevent overfitting.

## Training Process

The data is split into training and validation sets using the `train_test_split` function from scikit-learn. The model is trained for 10 epochs with a batch size of 64. Early stopping is applied with a patience of 3 to monitor the validation loss and restore the best weights.

## Evaluation Metrics

The model is evaluated on a separate test set. The following evaluation metrics are used:

- **Test Loss:** A measure of the model's performance on the test set.
- **Test Accuracy:** The proportion of correctly classified instances in the test set.

Additionally, a confusion matrix and ROC curve are plotted to provide a more detailed view of the model's performance, including true positive, true negative, false positive, and false negative rates.

## Challenges

1. **Data Cleaning:** Ensuring proper text cleaning and preprocessing to remove noise and irrelevant information from the reviews.
   
2. **Model Complexity:** Adjusting hyperparameters and the model architecture to prevent overfitting and achieve a balance between complexity and performance.

3. **Training Time:** Transformer models can be computationally expensive to train, and finding the right balance between training time and model performance is crucial.

## Conclusion

The presented Transformer model demonstrates competitive performance in sentiment analysis on the IMDb dataset. The use of early stopping helps prevent overfitting, and the evaluation metrics provide insights into the model's accuracy and generalization capabilities. The challenges faced during implementation highlight the importance of fine-tuning both the data preprocessing steps and model architecture for optimal results.
