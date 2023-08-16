Bank Transaction Categorization
A simple and powerful tool that uses a neural network to categorize bank transaction descriptions into respective categories and subcategories.

Overview
This program is designed to help with the categorization of bank transaction descriptions. It leverages the power of the BERT model to recognize patterns in transaction descriptions and subsequently classify them into predefined categories and subcategories.

Features
Data preprocessing to clean and prepare the transaction descriptions for the neural network.
Utilizes the BERT model, a state-of-the-art NLP model, for accurate categorization.
Simultaneous categorization into main category and subcategory.
Easy to use interface with simple Python functions.

How to Use
Setup & Installation

Make sure you have all the necessary libraries installed. Most importantly, ensure that torch, transformers, and pandas are available.

bash
Copy code
pip install torch transformers pandas

Code Structure
Data Preprocessing: DataPreprocessor class is responsible for reading the CSV file, cleaning the data, tokenizing the sentences, and preparing the data for training.
BERT Model: BertModel class initializes the BERT model with the appropriate number of categories and subcategories.
Training: train_category_model function handles the training loop, including batching, forward and backward passes, and early stopping.

Future Improvements
Implementing the subcategory model training function.
Refining the data preprocessing pipeline.
Fine-tuning the BERT model for better performance.
Implementing a gui for ease of use.
