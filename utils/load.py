import pandas as pd
from dicts import categories
from model import BertModel
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Load_Models:
    def __init__(self):
        """Initialize with the number of categories and subcategories."""
        self.category_keys = list(categories.keys())
        self.category_values = [item for sublist in categories.values() for item in sublist]
        self.num_categories = 0
        self.num_subcategories = 0
        self._get_num_categories()
    
    def _get_num_categories(self):

        # Convert categorical variables to numerical labels
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
        print("Number of categories:", self.num_categories)
        print("Number of subcategories:", self.num_subcategories)
    
    def load_model(self, model_load_path):
        """Load a BERT model from the specified path."""
        # Create the BERT model with the specified number of categories and subcategories
        model = BertModel(self.num_categories, self.num_subcategories)
        # Load the saved model weights
        state_dict = torch.load(model_load_path)
        model.load_state_dict(state_dict)
        # Set the model to evaluation mode
        model.eval()
        return model
    
    def merge_csv_files(self, *csv_files):
        # Check if the number of csv files is between 2 and 8
        if len(csv_files) < 2 or len(csv_files) > 8:
            print("Error: Invalid number of CSV files.")
            return None
        # Read the first csv file and create the dataframe
        df = pd.read_csv(csv_files[0])
        # Remove all columns except 'Description', 'Category', and 'Sub_Category'
        df = df[['Description', 'Category', 'Sub_Category']]
        # Combine remaining csv files to the dataframe
        for file in csv_files[1:]:
            temp_df = pd.read_csv(file, header=0)[['Description', 'Category', 'Sub_Category']]
            df = pd.concat([df, temp_df], ignore_index=True)
        df.dropna(subset=['Category', 'Sub_Category'], inplace=True)
        print("Successfully merged all CSV files into one dataframe.")
        return df