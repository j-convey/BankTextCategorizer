import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from data_prep import DataPreprocessor


class Model_Initializer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.obj = DataPreprocessor(filepath)
        self.num_categories = 0
        self.num_subcategories = 0

        self._get_num_categories()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
    
    def _get_num_categories(self):
        """Return number of categories."""
        self.num_categories, self.num_subcategories = self.obj.get_cat_sub_numbers()

    def get_data_loaders(self, batch_size):
        """Create and return data loaders for categories and subcategories."""
        self.obj.pop_columns(), self.obj.clean_dataframe(), self.obj.tokenize_data()

        (X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test) = self.obj.prepare_data()
        df = self.obj.get_df()

        # Convert to numpy arrays and argmax
        y_cat_train = np.array(y_cat_train)
        y_sub_train = np.array(y_sub_train)
        y_cat_test = np.array(y_cat_test)
        y_sub_test = np.array(y_sub_test)
        y_cat_train = np.argmax(y_cat_train, axis=1)
        y_sub_train = np.argmax(y_sub_train, axis=1)
        y_cat_test = np.argmax(y_cat_test, axis=1)
        y_sub_test = np.argmax(y_sub_test, axis=1)
        
        # Convert to tensors
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

        return cat_train_dataloader, cat_val_dataloader, sub_train_dataloader, sub_val_dataloader

