from transformers import BertTokenizer
from dicts import categories
from data_prep import DataPreprocessor
import torch
from torch.utils.data import TensorDataset, DataLoader

from load import Load_Models
from predict import Predict


def main():
   
    # Paths
    category_model_path = '/Users/Jordan Convey/Documents/GitHub/BankTextCategorizer/models/pt_cat_modelV1'
    csv_output_name = "fastftest.CSV"

    # Load model
    loader = Load_Models()
    # Load the category model using the load_model() method
    loaded_category_model = loader.load_model(category_model_path) 
    

    data_obj = DataPreprocessor('data/Standalones/Food - Fast Food.csv')
    predict_dataloader = data_obj.prepare_DATA()


    # Use load_models class to load the category model
    # model = Predict(loaded_category_model, predict_dataloader, csv_output_name)
    
    # print("Model loaded successfully")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)

    # model.run_prediction()

if __name__ == "__main__":
    main()