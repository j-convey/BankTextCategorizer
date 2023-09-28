import csv
import pandas as pd
from PyQt6.QtCore import pyqtSignal, QObject
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# custom modules
from data_prep import DataPreprocessor
from model import BertModel
from dicts import categories


class LogicHandler(QObject):
    progress_signal = pyqtSignal(int)
    def __init__(self):
        super(LogicHandler, self).__init__()
        self.cat_model_path = 'models/pt_cat_modelV1'
        self.sub_model_path = 'models/pt_sub_modelV1'
        self.combined_data = None
        self.category_keys = list(categories.keys())
        self.predict_df = pd.DataFrame()
        self.category_values = [item for sublist in categories.values() for item in sublist]
        self.num_categories = len(self.category_keys)
        self.num_subcategories = len(self.category_values)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder_cat = LabelEncoder()
        self.label_encoder_subcat = LabelEncoder()
        self.onehot_encoder_cat = OneHotEncoder(sparse_output=False)
        self.onehot_encoder_subcat = OneHotEncoder(sparse_output=False)
        self.integer_encoded_cat = self.label_encoder_cat.fit_transform(self.category_keys)
        self.onehot_encoded_cat = self.onehot_encoder_cat.fit_transform(self.integer_encoded_cat.reshape(-1, 1))
        # Encode category_values using label_encoder_subcat
        self.integer_encoded_subcat = self.label_encoder_subcat.fit_transform(self.category_values)
        self.onehot_encoded_subcat = self.onehot_encoder_subcat.fit_transform(self.integer_encoded_subcat.reshape(-1, 1))
        # Create dictionaries for category and sub-category mapping
        self.category_mapping = dict(zip(self.category_keys, self.onehot_encoded_cat))
        self.subcategory_mapping = dict(zip(self.category_values, self.onehot_encoded_subcat))
        self.num_categories = len(self.category_keys)
        # Number of subcategory
        self.num_subcategories = len(self.subcategory_mapping.keys())

    def return_processed_data(self):
        return self.predict_df
    
    def view_data(self):
        return self.combined_data
    
    def load_cat_model(self):
        cat_model = BertModel(self.num_categories, self.num_subcategories)
        state_dict = torch.load(self.cat_model_path)
        cat_model.load_state_dict(state_dict)
        cat_model.eval()
        return cat_model

    def load_sub_model(self):
        sub_model = BertModel(self.num_categories, self.num_subcategories)
        # Load saved model weights
        state_dict = torch.load(self.sub_model_path)
        sub_model.load_state_dict(state_dict)
        sub_model.eval()
        return sub_model

    def merge_csv_files(self, csv_files):
        if len(csv_files) < 2 or len(csv_files) > 8:
            raise ValueError("Select between 2 to 8 CSV files for merging.")
        df = pd.read_csv(csv_files[0])
        df = df[['Description', 'Category', 'Sub_Category']]
        for file in csv_files[1:]:
            temp_df = pd.read_csv(file, header=0)[['Description', 'Category', 'Sub_Category']]
            df = pd.concat([df, temp_df], ignore_index=True)
        self.combined_data = df
        return df
    
    def prep_df(self, df):
        # print df type
        print(type(df))
        df_obj = DataPreprocessor(df)
        df_obj.clean_dataframe()
        X_predict = df_obj.tokenize_predict_data()
        predict_input_ids = torch.tensor(X_predict, dtype=torch.long)
        predict_dataset = TensorDataset(predict_input_ids)
        predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
        print("Length of predict_dataloader:", len(predict_dataloader))
        return predict_dataloader

    def predict(self, df, progress_callback=None):
        # Load models
        cat_model = self.load_cat_model()
        sub_model = self.load_sub_model()
        cat_model.to(self.device)
        sub_model.to(self.device)
        output_csv = 'predict_data_test.csv'
        predict_dataloader = self.prep_df(df)
        total_items = len(predict_dataloader.dataset)
        processed_items = 0
        # Lists to store predictions and descriptions
        categories, subcategories, descriptions = [], [], []
        for batch in predict_dataloader:
            input_ids = batch[0].to(self.device)
            with torch.no_grad():
                category_probs, _ = cat_model(input_ids)
                category_predictions = category_probs.argmax(dim=-1)
                _, subcategory_probs = sub_model(input_ids)
                subcategory_predictions = subcategory_probs.argmax(dim=-1)
            processed_items += 1
            progress_percentage = int((processed_items / total_items) * 100)
            self.progress_signal.emit(progress_percentage)
            for i in range(input_ids.size(0)):
                category_name = self.label_encoder_cat.inverse_transform([category_predictions[i].item()])[0]
                categories.append(category_name)
                subcategory_name = self.label_encoder_subcat.inverse_transform([subcategory_predictions[i].item()])[0]
                subcategories.append(subcategory_name)
                # Unmap input_ids to the original description
                single_input_ids = input_ids[i].to('cpu')
                tokens = self.tokenizer.convert_ids_to_tokens(single_input_ids)
                description = self.tokenizer.convert_tokens_to_string([token for token in tokens if token != "[PAD]"]).strip()
                descriptions.append(description)  
                 
        # Write to CSV
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Description', 'Category', 'Subcategory']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            for description, category, subcategory in zip(descriptions, categories, subcategories):
                writer.writerow({'Description': description, 'Category': category, 'Subcategory': subcategory})
        self.predict_df = pd.read_csv(output_csv)
        return output_csv