import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from dicts import categories
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from data_prep import DataPreprocessor
from model import BertModel
import torch
from torch.utils.data import TensorDataset, DataLoader


category_keys = list(categories.keys())
category_values = [item for sublist in categories.values() for item in sublist]

def load_category_model(model_load_path):
    # Define the category model architecture
    model = BertModel(num_categories, num_subcategories)  # No subcategories in this model
    # Load the saved model weights
    state_dict = torch.load(model_load_path)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def load_subcategory_model(model_load_path):
    # Define the subcategory model architecture
    model = BertModel(0, num_subcategories)  # No categories in this model
    # Load the saved model weights
    state_dict = torch.load(model_load_path)
    # Load the model state dictionary into the model architecture
    model.load_state_dict(state_dict)
    # Set the model to evaluation mode
    model.eval()
    return model

def merge_csv_files(*csv_files):
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

def predict(loaded_category_model, predict_dataloader, device, csv_output_name):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    loaded_category_model.to(device)
    #subcategory_model.to(device)
    with open(csv_output_name, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Description', 'Category', 'Subcategory']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for batch in predict_dataloader:
            input_ids = batch[0].to(device)
            with torch.no_grad():
                category_probs, _ = loaded_category_model(input_ids)
                #_, subcategory_probs = subcategory_model(input_ids)
                category_predictions = category_probs.argmax(dim=-1)
                #subcategory_predictions = subcategory_probs.argmax(dim=-1)
            for i in range(input_ids.size(0)):
                category_name = label_encoder_cat.inverse_transform([category_predictions[i].item()])[0]
                #subcategory_name = label_encoder_subcat.inverse_transform([subcategory_predictions[i].item()])[0]
                # Unmap input_ids to the original description
                single_input_ids = input_ids[i].to('cpu')
                tokens = tokenizer.convert_ids_to_tokens(single_input_ids)
                description = tokenizer.convert_tokens_to_string(tokens).strip()
                writer.writerow({'Description': description, 'Category': category_name})
    return None

if __name__ == "__main__":
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
    category_mapping = dict(zip(category_keys, onehot_encoded_cat))
    subcategory_mapping = dict(zip(category_values, onehot_encoded_subcat))
    num_categories = len(category_keys)
    # Number of subcategory
    num_subcategories = len(subcategory_mapping.keys())
    #data_path = ('PREPARED FOR TRAINING.csv')
    category_model_path = '/Users/Jordan Convey/Documents/GitHub/BankTextCategorizer/models/pt_cat_modelV1'
    csv_output_name = "fastftest.CSV"
    #merged_df.to_csv("PREPARED FOR TRAINING.csv", index=False)
    books_obj = DataPreprocessor('data/Standalones/Food - Fast Food.csv')
    books_obj.clean_dataframe()
    books_obj.tokenize_predict_data()
    X_predict = books_obj.tokenize_predict_data()
    predict_input_ids = torch.tensor(X_predict, dtype=torch.long)
    predict_dataset = TensorDataset(predict_input_ids)
    predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
    print("Length of predict_dataloader:", len(predict_dataloader))
    loaded_category_model = load_category_model(category_model_path)
    #loaded_subcategory_model = load_subcategory_model(subcategory_model_path)
    print("Model loaded successfully")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    predict(loaded_category_model, predict_dataloader, device, csv_output_name)

    