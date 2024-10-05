My approach to this ML challenge was heavily influenced by my hardware limitations since I have a RTX 3060 mobile GPU to work with I couldn't choose anylarge models escpecially before the submission date was extendend therefore I decided on a solution which would combine 4 smaller models to get the maximum possible score despite my hardware limitations.

I basically finetuned 4 transformer models on the datset generated their indivudual submission files and also generated their probablilities which I then used in ensemble to train a meta model - XGBoost on these probabilities this way the XGBoost model would learn from the other models and give me a further boost in accuracy.

For this meta model to give a higher score the models would have to have different architectures so after a lot of trial and error, I settled on 4 transformer based models.


Project Approach with Feature Engineering and Tools:



1. Data Preprocessing and Feature Engineering:
Feature engineering plays a crucial role in transforming the raw text data into a format that can be fed into machine learning models. Below are the specific steps taken for feature engineering and the tools used:

a. Text Preprocessing:
Lowercasing: The text of the articles is converted to lowercase using pandas str.lower() function to ensure case-insensitive comparisons and feature extraction.
Whitespace Stripping: Extra whitespaces are removed from the text using the str.strip() function to normalize input.
Tokenization: This step is critical to converting text into numerical format. Tokenization breaks down each article into individual tokens (words or subwords) using tokenizers provided by pre-trained models (DeBERTa, Electra, MiniLM, etc.).
For instance, the DeBERTaTokenizer, ElectraTokenizer, and BertTokenizer are used to transform the text data into tokens, which represent the input features for each model.
Padding and Truncation: Since articles vary in length, we use padding and truncation to ensure all inputs are of uniform length. A maximum sequence length of 512 tokens is specified to truncate longer texts and pad shorter ones.

b. Label Encoding:
The categorical target labels (IAB categories) are converted into numeric format using the LabelEncoder from Scikit-learn. This converts each category into a unique integer that the models can use for classification.
For example, categories like 'Academic Interests', 'Technology News', and others are converted into integers such as 0, 1, 2, ... n.

c. Tokenization as Feature Engineering:
The Hugging Face library is used for tokenization. This converts the raw text into input IDs, attention masks, and other input features that are required for transformer models.
Attention Masks: These are used to differentiate between real tokens and padding tokens.
Hugging Face Datasets: After tokenization, the processed data is stored in Hugging Face’s Dataset format, which efficiently handles large datasets and integrates well with transformer models.



2. Transformer Models and Feature Extraction:
Each pre-trained transformer model used (DeBERTa, MiniLM, Electra, BERT) acts as a feature extractor by encoding the input text into rich embeddings, which are then used for classification. Here are the tools and techniques used for each model:

a. DeBERTa v3 XSmall:
Tokenizer: DebertaTokenizer breaks down the input text into tokens.
Model: AutoModelForSequenceClassification is used to fine-tune the model on the classification task. It extracts features from the tokenized text and passes them through its transformer layers.
FP16 Precision: Mixed precision (16-bit floating point) is used to optimize memory usage and training speed, particularly when using GPU.

b. Electra Small:
Discriminator Model: The google/electra-small-discriminator model is fine-tuned for sequence classification, which outputs probabilities for each class.
Feature Extraction: The tokenized input is passed through Electra’s transformer layers to extract high-dimensional features that are used to predict IAB categories.

c. MiniLM:
MiniLM Tokenizer: Tokenizes the text data into subword units.
MiniLM Model: The MiniLM-L12-H384-uncased model is used, which is a lightweight model that provides fast training and inference times while maintaining high accuracy.
Feature Extraction: The model processes the tokenized inputs to generate embeddings, which are then classified into the respective categories.

d. BERT (Custom Fine-tuned Model):
BERT Tokenizer: Tokenizes the input text to prepare it for the model.
BERT Base Model: The pre-trained bert-base-uncased model fine-tuned on IAB categories is used here. During fine-tuning, most layers are frozen, and only the last few transformer layers are allowed to learn, reducing training time and preventing overfitting.
Feature Extraction: The model encodes the input text into embeddings that represent the article's semantic content.



3. Model Training and Optimization Tools:

a. Trainer API (Hugging Face):
The Trainer class from Hugging Face provides an easy-to-use interface for training and fine-tuning transformer models.
It handles everything from batching to gradient accumulation and logging.
We define custom compute_metrics functions for evaluation, using the F1 score and accuracy as key performance metrics.

b. GPU Acceleration and Mixed Precision:
The training of models is accelerated using GPUs, and mixed precision (FP16) is enabled to reduce memory usage and speed up training.
For training with GPUs, we set up torch.device("cuda") to leverage the GPU, and the TrainingArguments include the fp16=True flag for mixed precision.

c. Early Stopping:
EarlyStoppingCallback is used to halt training once the validation performance stops improving for a set number of steps (patience). This prevents overfitting and saves computation resources.



4. Ensembling and Meta-Model:

a. Predicted Probabilities as Features:
After training each model, we save the predicted probabilities on both the training set and the test set. These probabilities are treated as new features (meta-features) for the next stage, where a meta-classifier is used to make final predictions.

b. XGBoost Meta-Classifier:
The predicted probabilities from DeBERTa and MiniLM (or additional models like Electra) are concatenated to form a matrix of features (meta-features).
XGBoost is used as the meta-classifier to ensemble these predictions.
We set up XGBoost with parameters optimized for classification tasks, such as max_depth, learning_rate, and n_estimators, while utilizing GPU acceleration for faster training.

c. Meta-Features Engineering:
The meta-features are created by hstacking (horizontally stacking) the predicted probabilities from each model.
This step creates a richer representation of the input data by combining the knowledge learned by each individual model.



5. Final Prediction and Submission:
The meta-classifier makes final predictions on the test set by combining the probabilities generated by the individual models.
These predictions are converted back to the original IAB categories using inverse transformation of the LabelEncoder.
The final predictions are saved in a CSV submission file.

Tools Used:
Hugging Face Transformers: Tokenization, feature extraction, and training.
Pandas: Data manipulation and loading.
NumPy: Numerical operations and saving probabilities.
Scikit-learn: Label encoding, train-validation splitting, and metrics calculation (accuracy, F1 score).
XGBoost: Meta-classifier for ensembling.
PyTorch: Model training and GPU acceleration.
CUDA: GPU acceleration for faster training and inference.

