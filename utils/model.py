import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertForSequenceClassification
import matplotlib.pyplot as plt
from data_prep import DataPreprocessor
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def init_model_data():
    # Data Processing
    books_obj = DataPreprocessor('data/main.csv')
    num_categories, num_subcategories = books_obj.get_cat_sub_numbers()
    books_obj.pop_columns(), books_obj.clean_dataframe(), books_obj.tokenize_data()
    (X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test) = books_obj.prepare_data()
    df = books_obj.get_df()
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
    batch_size = 64
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
    cat_model = BertModel(num_categories, num_subcategories)
    sub_model = BertModel(num_categories, num_subcategories)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return cat_model, sub_model, cat_train_dataloader, cat_val_dataloader, sub_train_dataloader, \
        sub_val_dataloader, device, num_categories, num_subcategories

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
    
def train_model(model, model_type, train_dataloader, val_dataloader, epochs, learning_rate, device, print_interval=1, patience=5):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    category_loss_fn = nn.CrossEntropyLoss(reduction='sum')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model = model.to(device)
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    batch_print_interval = 5
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        for i, batch in enumerate(train_dataloader):
            input_ids, y_cat = [item.to(device) for item in batch[:2]]
            optimizer.zero_grad()
            if model_type == 'category':
                cat_probs, _ = model(input_ids)
            elif model_type == 'subcategory':
                _, cat_probs = model(input_ids)
            cat_loss = category_loss_fn(cat_probs, y_cat)
            total_train_loss += cat_loss.item()
            correct_train += (cat_probs.argmax(dim=1) == y_cat).sum().item()
            cat_loss.backward()
            optimizer.step()
            if (i + 1) % batch_print_interval == 0:
                # Validation phase inside the batch loop
                model.eval()
                total_val_loss = 0
                correct_val = 0        
                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids, y_cat = [item.to(device) for item in batch[:2]]
                        if model_type == 'category':
                            cat_probs, _ = model(input_ids)
                        elif model_type == 'subcategory':
                            _, cat_probs = model(input_ids)
                        cat_loss = category_loss_fn(cat_probs, y_cat)                
                        total_val_loss += cat_loss.item()
                        correct_val += (cat_probs.argmax(dim=1) == y_cat).sum().item()        
                avg_val_loss = total_val_loss / len(val_dataloader)
                val_acc = correct_val / len(val_dataloader.dataset)
                avg_train_loss = total_train_loss / (i + 1)  # current average for this epoch up to batch i
                train_acc = (correct_train / ((i + 1) * len(batch))) / 100
                print(f"Epoch {epoch}/{epochs} - Batch {i+1}/{len(train_dataloader)} "
                    f"- Training loss: {avg_train_loss:.4f}, Training Acc: {train_acc:.4f}, "
                    f"Validation loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.4f}")
                model.train()  # Switch back to training mode
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        # Update learning rate
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1           
        if no_improvement_epochs >= patience:
            print(f"Stopping early due to no improvement after {patience} epochs.")
            break
    return history

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
    learning_rate = 1e-5
    epochs = 2
    cat_model, sub_model, cat_train_dataloader, cat_val_dataloader, \
    sub_train_dataloader, sub_val_dataloader, device, num_categories, num_subcategories = init_model_data()
    # Execute & Save Models
    #execute_cat_model(cat_model, cat_train_dataloader, cat_val_dataloader, device, num_categories, learning_rate, epochs)
    execute_sub_model(sub_model, sub_train_dataloader, sub_val_dataloader, device, num_subcategories, learning_rate, epochs)
if __name__ == '__main__':
    main()
