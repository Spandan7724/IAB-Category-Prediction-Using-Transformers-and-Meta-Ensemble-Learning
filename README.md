# IAB Category Prediction Using Transformers and Meta-Ensemble Learning

This project aims to build an effective machine learning pipeline to classify articles into predefined IAB (Interactive Advertising Bureau) categories using several transformer models like DeBERTa, MiniLM, Electra, and BERT. The ensemble approach leverages the strengths of these models by combining them in a meta-learning framework to achieve higher accuracy and generalization.

## Project Overview

The objective of this project is to predict IAB categories for a given dataset of news articles. The project includes:

-   Training multiple state-of-the-art transformer models.
-   Implementing feature engineering techniques for text preprocessing.
-   Using a meta-classifier (XGBoost) to improve prediction accuracy by combining model outputs.
-   Providing a comprehensive pipeline from data preprocessing, model training, to final prediction.

## Dataset

The dataset consists of two CSV files:

-   `train.csv`: Contains the training data with the text of the articles and their corresponding IAB category labels.
-   `test.csv`: Contains the test data with the text of the articles for which the categories need to be predicted.

The dataset includes the following columns:

-   **Index**: A unique identifier for each article.
-   **Text**: The actual content of the article.
-   **Target**: The IAB category label corresponding to each article (only in `train.csv`).

## Approach

### Feature Engineering and Preprocessing

-   **Text Normalization**: The text data was converted to lowercase and stripped of any leading/trailing whitespace.
-   **Label Encoding**: The target labels (IAB categories) were transformed into numerical labels using `LabelEncoder` for model compatibility.

### Model Training

Several transformer models were used for sequence classification tasks:

-   **DeBERTa (v3-xsmall)**: A transformer model from Microsoft focused on better handling of disambiguation in text.
-   **Electra (small-discriminator)**: A transformer-based model that trains to detect replaced tokens.
-   **MiniLM (L12-H384)**: A lightweight version of BERT optimized for faster inference.
-   **BERT (Base-uncased)**: A pre-trained BERT model fine-tuned specifically for text classification tasks.

The models were trained using Hugging Face’s `Trainer` API with the following features:

-   Mixed precision (`fp16`) training for faster computation.
-   Early stopping to prevent overfitting.
-   Batch size tuning based on GPU memory availability.

### Meta-Classifier

After training the individual models, an ensemble technique was employed:

-   **XGBoost Meta-Classifier**: A meta-classifier was trained on the predicted probabilities from the individual models (DeBERTa and MiniLM). These predicted probabilities were used as meta-features for the XGBoost classifier, which predicted the final class labels.

### Evaluation Metrics

-   **Accuracy**: The percentage of correctly classified articles.
-   **F1-Score**: The harmonic mean of precision and recall, computed with weighted averaging.

## Tools and Libraries Used

-   **Transformers**: Hugging Face’s Transformers library for loading pre-trained models and fine-tuning.
-   **Datasets**: Hugging Face’s Datasets library for efficient data processing.
-   **XGBoost**: A gradient boosting library used for the meta-classifier.
-   **Pandas**: For data loading and manipulation.
-   **Sklearn**: For preprocessing, metrics, and model evaluation.
-   **PyTorch**: For model training and GPU acceleration.

## Directory Structure
`
`├── dataset_fibe
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
├── deberta
│   ├── saved_model
│   ├── saved_tokenizer
│   ├── tokenized_train_dataset
│   ├── tokenized_val_dataset
│   ├── tokenized_test_dataset
├── electra
│   ├── saved_model
│   ├── saved_tokenizer
│   ├── tokenized_train_dataset
│   ├── tokenized_val_dataset
│   ├── tokenized_test_dataset
├── minilm
│   ├── saved_model
│   ├── saved_tokenizer
│   ├── tokenized_train_dataset
│   ├── tokenized_val_dataset
│   ├── tokenized_test_dataset
├── nofin
│   ├── saved_model
│   ├── saved_tokenizer
│   ├── tokenized_train_dataset
│   ├── tokenized_val_dataset
│   ├── tokenized_test_dataset
├── meta_classifier_submission.csv
├── meta_classifier.py  # Meta-classifier training and evaluation script
├── README.md

## Results

The ensemble of DeBERTa, MiniLM, and XGBoost meta-classifier provides a robust model that achieves high accuracy and F1 scores on the test set. The final model uses predicted probabilities from individual models as inputs to the meta-classifier, which produces the final prediction.

## Dataset-

https://drive.google.com/drive/folders/1IpQmp0782I6Fkz1FIz54z8GSSNHyEd83?usp=sharing

## Future Work

-   Add more transformer models to the ensemble to improve performance.
-   Perform hyperparameter tuning of the XGBoost meta-classifier for better generalization.
-   Experiment with alternative meta-learners, such as CatBoost or LightGBM.
