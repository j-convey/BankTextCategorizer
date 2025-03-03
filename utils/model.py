import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainer import train_model, train_and_save_model
from init_model import Model_Initializer 

class BertModel(nn.Module):
    def __init__(self, num_categories, num_subcategories):
        super().__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels= num_categories + num_subcategories)      
        self.num_categories = num_categories
        self.num_subcategories = num_subcategories
        
    def forward(self, input_ids):
        outputs = self.bert_model(input_ids)
        logits = outputs.logits
        category_logits, subcategory_logits = logits.split([self.num_categories, self.num_subcategories], dim=-1)
        return category_logits, subcategory_logits
    

def plot_training_history(history):
    expected_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
    for key in expected_keys:
        if key not in history.keys():
            print(f"Error: Expected key {key} not found in history")
            return
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def execute_cat_model(cat_model, cat_train_dataloader, cat_val_dataloader, device, num_categories, learning_rate, epochs):
    '''Category Training & Saving'''    
    cat_model.to(device)
    category_history = train_model(cat_model, cat_train_dataloader, cat_val_dataloader, epochs, learning_rate, device, num_categories)
    # Move the model back to CPU before saving
    cat_model.to('cpu')
    cat_model_save_path = 'models/pt_cat_modelV1'
    torch.save(cat_model.state_dict(), cat_model_save_path)
    plot_training_history(category_history)

def execute_sub_model(sub_model, sub_train_dataloader, sub_val_dataloader, device, num_subcategories, learning_rate, epochs):
    '''Subcategory Training & Saving'''
    sub_model.to(device)
    subcategory_history = train_model(sub_model, 'subcategory', sub_train_dataloader, sub_val_dataloader, epochs, learning_rate, device)
    sub_model.to('cpu')
    sub_model_save_path = 'models/pt_sub_modelV1'
    torch.save(sub_model.state_dict(), sub_model_save_path)
    plot_training_history(subcategory_history)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    learning_rate = 1e-5
    epochs = 2
    batch_size = 64

    data_obj = Model_Initializer('data/main.csv')

    cat_train_dataloader, cat_val_dataloader, sub_train_dataloader, sub_val_dataloader = data_obj.get_data_loaders(batch_size=64)
    num_categories = data_obj.num_categories
    num_subcategories = data_obj.num_subcategories

    # Model initialization
    cat_model = BertModel(num_categories, num_subcategories)
    sub_model = BertModel(num_categories, num_subcategories)



    # Train and save models
    # Category model (uncomment to enable)
    # cat_history = train_and_save_model(cat_model, 'category', cat_train_dataloader, cat_val_dataloader, 
    #                                    'models/pt_cat_modelV1', epochs, learning_rate, device)
    # plot_training_history(cat_history)
    
    # Subcategory model
    sub_history = train_and_save_model(sub_model, 'subcategory', sub_train_dataloader, sub_val_dataloader, 
                                       'models/pt_sub_modelV1', epochs, learning_rate, device)
    plot_training_history(sub_history)
if __name__ == '__main__':
    main()
