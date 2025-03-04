import re
import pandas as pd
import random
import torch
import numpy as np
from transformers import BertTokenizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

from dicts import categories
from model import BertModel
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 50)
stop_words = set(stopwords.words('english'))

class DataPreprocessor:
    '''Data Preparation: Prepare a dataset of bank transaction descriptions along with their corresponding categories and sub-categories.'''
    def __init__(self, data_input):
        # If it's a string, assume it's a file path and read it into a DataFrame
        if isinstance(data_input, str):
            self.df = pd.read_csv(data_input)
            self.file_name = data_input.split('/')[-1].split('.')[0]
        elif isinstance(data_input, pd.DataFrame):
            self.df = data_input
            self.file_name = ""  # Default to an empty string or set it to some value if needed
        else:
            raise ValueError("data_input must be a file path or a pandas DataFrame")


        self.category_keys = list(categories.keys())
        self.category_values = [item for sublist in categories.values() for item in sublist]
        self.num_categories = 0
        self.num_subcategories = 0
        self.category_mapping = {}
        self.subcategory_mapping = {}

        self._category_nums()

    def get_df(self):
        return self.df
    
    def _category_nums(self):
        label_encoder_cat = LabelEncoder()
        label_encoder_subcat = LabelEncoder()
        onehot_encoder_cat = OneHotEncoder(sparse_output=False)
        onehot_encoder_subcat = OneHotEncoder(sparse_output=False)
        # Encode category_keys using label_encoder_cat
        integer_encoded_cat = label_encoder_cat.fit_transform(self.category_keys)
        onehot_encoded_cat = onehot_encoder_cat.fit_transform(integer_encoded_cat.reshape(-1, 1))
        # Encode category_values using label_encoder_subcat
        integer_encoded_subcat = label_encoder_subcat.fit_transform(self.category_values)
        onehot_encoded_subcat = onehot_encoder_subcat.fit_transform(integer_encoded_subcat.reshape(-1, 1))
        # Create dictionaries for category and sub-category mapping
        self.category_mapping = dict(zip(self.category_keys, onehot_encoded_cat))
        self.subcategory_mapping = dict(zip(self.category_values, onehot_encoded_subcat))
        # Number of category
        self.num_categories = len(self.category_keys)
        # Number of subcategory
        self.num_subcategories = len(self.subcategory_mapping.keys())

    def get_cat_sub_numbers(self):
        return self.num_categories, self.num_subcategories
    
    def compute_class_weights(self, labels, num_classes):
        class_counts = np.bincount(labels, minlength=num_classes)
        class_frequencies = (class_counts + 1) / (np.sum(class_counts) + num_classes)
        class_weights = 1 / class_frequencies
        if np.sum(class_weights) == 0:
            return torch.ones(num_classes, dtype=torch.float32) / num_classes
        else:
            normalized_weights = class_weights
            return torch.tensor(normalized_weights, dtype=torch.float32)

    def tokenize_data(self, max_len=50):
        # Load the BERT Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Tokenize the 'Description' column
        tokenized_desc = [tokenizer.tokenize(text) for text in self.df['Description']]
        self.df['Tokenized'] = tokenized_desc
        # Convert the tokenized sequences to IDs
        tokenized_desc_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_desc]
        self.df['Tokenized_ids'] = tokenized_desc_ids
        # Find the maximum length of the tokenized sequences
        actual_max = max([len(tokens) for tokens in tokenized_desc_ids])
        print("Max length in the data:", actual_max)
        # Pad the sequences to the max_len using NumPy
        padded_desc = np.array([seq[:max_len] + [0] * (max_len - len(seq)) if len(seq) < max_len 
                            else seq[:max_len] for seq in tokenized_desc_ids])
        self.df['Tokenized_padded'] = padded_desc.tolist()
        # Map the Category and Sub_Category values to the corresponding one-hot encoded vectors
        self.df['Tok_Cat'] = self.df['Category'].apply(lambda x: self.category_mapping.get(x))
        self.df['Tok_Sub'] = self.df['Sub_Category'].apply(lambda x: self.subcategory_mapping.get(x))
        return self.df

    def prepare_data(self):
        self.df = self.df.dropna(subset=["Sub_Category", "Category"])
        # Separate Category and Sub_Category labels
        y_cat = self.df["Tok_Cat"].tolist()
        y_sub = self.df["Tok_Sub"].tolist()
        X = self.df["Tokenized_padded"].tolist()
        X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test = train_test_split(X, y_cat, y_sub, test_size=0.33, random_state=42)
        return (X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test)

    def pop_columns(self):
        self.df['Category'] = self.df['Category'].astype(str)
        # Add new columns
        self.df = self.df.assign(Tokenized='', Tok_Cat='', Tok_Sub='')
        rename_columns = {'Product Name': 'Description', 'Category': 'Category', 'Vector': 'Vector'}
        col_order = ['Description', 'Category', 'Sub_Category', 'Tok_Cat', 'Tok_Sub', 'Tokenized']
        # Create a list of columns to keep
        keep_columns = [col for col in rename_columns.keys()] + [col for col in rename_columns.values()] + ['Tokenized', 'Sub_Category', 'Tok_Cat', 'Tok_Sub']
        # Rename columns that match the keys in the dictionary
        self.df = self.df.rename(columns=rename_columns)
        if 'Sub_Category' not in self.df.columns:
            self.df = self.df.assign(Sub_Category='')
        # Reorder the columns
        self.df = self.df.reindex(columns=col_order)
        # Drop columns that don't match the keys or values in the dictionary
        self.df = self.df.drop([col for col in self.df.columns if col not in keep_columns], axis=1)
        # Make sure all columns aren't empty
        self.df.dropna(subset=["Description"], axis=0, inplace=True)
        self.df.fillna({"Category": ""}, inplace=True)
        return self.df

    def clean_dataframe(self):
        #Remove any duplicate rows where Description and Category are the same
        self.df.drop_duplicates(subset=['Description', 'Category'], keep='first', inplace=True)
        # Use regular expression to remove non-letter characters
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', str(x)))
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z\-\' ]', '', x))
        #remove white spaces and make all lowercase
        self.df['Description'] = self.df['Description'].apply(lambda x: x.strip().lower())
        # Replace multiple spaces with a single space
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'\s+', ' ', x))
        #Remove any numbers from the description
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'\d+', '', x))
        #Remove the word 'and', 'the' from the description
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'\b(and|the|set|inch|of|in|made|to|by|compatible|with|for|set|other|cm|st|street|ave)\b', '', x))
        #Make all letters lowercase in Description column
        self.df['Description'] = self.df['Description'].str.lower()
        #Remove any single letter words from Description column
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'\b[a-zA-Z]\b', '', x))
        #Remove stop words from Description column using NLTK
        self.df['Description'] = self.df['Description'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        #self.df.dropna(subset=['Category', 'Sub_Category'], inplace=True)
        return self.df

    def clean_predict_data(self):
        self.df["Category"].fillna("", inplace=True)
        self.df["Sub_Category"].fillna("", inplace=True)
        #Remove any duplicate rows where Description and Category are the same
        self.df.drop_duplicates(subset=['Description', 'Category'], keep='first', inplace=True)
        # Use regular expression to remove non-letter characters
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', x))    
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z\-\' ]', '', x))
        #remove white spaces and make all lowercase
        self.df['Description'] = self.df['Description'].apply(lambda x: x.strip().lower())
        # Replace multiple spaces with a single space
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'\s+', ' ', x))
        #Remove any numbers from the description
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'\d+', '', x))
        #Remove the word 'and', 'the' from the description
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'\b(and|the|set|inch|of|in|made|to|by|compatible|with|for|set|other|cm|st|street|ave)\b', '', x))
        #Make all letters lowercase in Description column
        self.df['Description'] = self.df['Description'].str.lower()
        #Remove any single letter words from Description column
        self.df['Description'] = self.df['Description'].apply(lambda x: re.sub(r'\b[a-zA-Z]\b', '', x))
        #Remove stop words from Description column using NLTK
        self.df['Description'] = self.df['Description'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        self.df.dropna(subset=['Category', 'Sub_Category'], inplace=True)
        return self.df
    
    def tokenize_predict_data(self, max_len=105):
        # Load the BERT Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Tokenize the 'Description' column
        tokenized_desc = [tokenizer.tokenize(text) for text in self.df['Description']]
        self.df['Tokenized'] = tokenized_desc
        # Convert the tokenized sequences to IDs
        tokenized_desc_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_desc]
        self.df['Tokenized_ids'] = tokenized_desc_ids
        # Find the maximum length of the tokenized sequences
        actual_max = max([len(tokens) for tokens in tokenized_desc_ids])
        print("Max length in the data:", actual_max)
        # Pad the sequences to the max_len using NumPy
        padded_desc = np.array([seq[:max_len] + [0] * (max_len - len(seq)) if len(seq) < max_len 
                            else seq[:max_len] for seq in tokenized_desc_ids])
        X = self.df['Tokenized_padded'] = padded_desc.tolist()
        # print(X)
        return X
    
    def predict_prepare_data(self):
        self.df = self.df.dropna(subset=["Sub_Category", "Category"])
        # Separate Category and Sub_Category labels
        X = self.df["Tokenized_padded"].tolist()
        return X
    
    def shuffle_sentences(self):
        """Double the dataset by shuffling the words in the Description."""
        augmented_df = self.df.copy()
        # Shuffle the words in each Description
        augmented_df['Description'] = augmented_df['Description'].apply(lambda desc: ' '.join(random.sample(desc.split(), len(desc.split()))))
        # Concatenate the original and augmented dataframes
        self.df = pd.concat([self.df, augmented_df], ignore_index=True)
        return self.df

    def prepare_DATA(self):
        self.clean_dataframe()
        self.tokenize_predict_data()
        X_predict = self.tokenize_predict_data()
        predict_input_ids = torch.tensor(X_predict, dtype=torch.long)
        predict_dataset = TensorDataset(predict_input_ids)
        predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
        print("Length of predict_dataloader:", len(predict_dataloader))
        return predict_dataloader


    def prepare_predict(self):
        category_keys = list(categories.keys())
        category_values = [item for sublist in categories.values() for item in sublist]
        # Convert categorical variables to numerical labels
        label_encoder_cat = LabelEncoder()
        label_encoder_subcat = LabelEncoder()
        onehot_encoder_cat = OneHotEncoder(sparse_output=False)
        onehot_encoder_subcat = OneHotEncoder(sparse_output=False)
        # Encode category_keys using label_encoder_cat
        integer_encoded_cat = label_encoder_cat.fit_transform(category_keys)
        onehot_encoded_cat = onehot_encoder_cat.fit_transform(integer_encoded_cat.reshape(-1, 1))
        # Encode category_values using label_encoder_subcat
        integer_encoded_subcat = label_encoder_subcat.fit_transform(category_values)
        onehot_encoded_subcat = onehot_encoder_subcat.fit_transform(integer_encoded_subcat.reshape(-1, 1))
        # Create dictionaries for category and sub-category mapping
        self.category_mapping = dict(zip(category_keys, onehot_encoded_cat))
        self.subcategory_mapping = dict(zip(category_values, onehot_encoded_subcat))
        # Number of category
        self.num_categories = len(category_keys)
        # Number of subcategory
        self.num_subcategories = len(self.subcategory_mapping.keys())

    def prepare_for_training(self, batch_size=64):
        self.pop_columns() 
        self.clean_dataframe()
        self.tokenize_data()
        (X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test) = self.prepare_data()
        df = self.get_df()
        y_cat_train = np.array(y_cat_train)
        y_sub_train = np.array(y_sub_train)
        y_cat_test = np.array(y_cat_test)
        y_sub_test = np.array(y_sub_test)
        y_cat_train = np.argmax(y_cat_train, axis=1)
        y_sub_train = np.argmax(y_sub_train, axis=1)
        y_cat_test = np.argmax(y_cat_test, axis=1)
        y_sub_test = np.argmax(y_sub_test, axis=1)
        # Now convert the numpy arrays to tensors
        y_cat_train = torch.tensor(y_cat_train, dtype=torch.long)
        y_sub_train = torch.tensor(y_sub_train, dtype=torch.long)
        y_cat_test = torch.tensor(y_cat_test, dtype=torch.long)
        y_sub_test = torch.tensor(y_sub_test, dtype=torch.long)
        # Convert tokenized sequences to input IDs
        train_input_ids = torch.tensor(X_train)
        val_input_ids = torch.tensor(X_test)
        print(f"Total number of data points: {df.shape[0]}")
        print(f"Number of training data points: {len(X_train)}")
        print(f"Number of testing data points: {len(X_test)}")
        
        # Category data loaders
        cat_train_dataset = TensorDataset(train_input_ids, y_cat_train)
        cat_val_dataset = TensorDataset(val_input_ids, y_cat_test)
        cat_train_dataloader = DataLoader(cat_train_dataset, batch_size=batch_size, shuffle=True)
        cat_val_dataloader = DataLoader(cat_val_dataset, batch_size=batch_size, shuffle=False)
        # Subcategory data loaders
        sub_train_dataset = TensorDataset(train_input_ids, y_sub_train)
        sub_val_dataset = TensorDataset(val_input_ids, y_sub_test)
        sub_train_dataloader = DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=True)
        sub_val_dataloader = DataLoader(sub_val_dataset, batch_size=batch_size, shuffle=False)
        print("Number of training batches:", len(cat_train_dataloader))
        print("Number of validation batches:", len(sub_val_dataloader))


        return cat_train_dataloader, cat_val_dataloader, sub_train_dataloader, sub_val_dataloader
