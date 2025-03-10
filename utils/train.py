import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_prep import DataPreprocessor
from model import BertModel
from utils import plot_training_history

class TrainModel:
    def __init__(self, model, model_type, train_dataloader, val_dataloader, epochs, learning_rate, patience=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_type = model_type
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        self.category_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_loss = float('inf')
        self.no_improvement_epochs = 0

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss, correct_train = 0, 0
            for i, batch in enumerate(self.train_dataloader):
                loss, correct = self._train_step(batch)
                total_train_loss += loss
                correct_train += correct
                if (i + 1) % 5 == 0:
                    avg_val_loss, val_acc = self.validate()
                    avg_train_loss = total_train_loss / (i + 1)
                    train_acc = (correct_train / ((i + 1) * len(batch))) / 100
                    print(f"Epoch {epoch}/{self.epochs} - Batch {i+1}/{len(self.train_dataloader)} "
                          f"- Training loss: {avg_train_loss:.4f}, Training Acc: {train_acc:.4f}, "
                          f"Validation loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.4f}")
            self._update_history(avg_train_loss, train_acc, avg_val_loss, val_acc)
            self._check_early_stopping(avg_val_loss)
            if self.no_improvement_epochs >= self.patience:
                print(f"Stopping early due to no improvement after {self.patience} epochs.")
                break
        return self.history

    def _train_step(self, batch):
        input_ids, y_cat = [item.to(self.device) for item in batch[:2]]
        self.optimizer.zero_grad()
        if self.model_type == 'category':
            cat_probs, _ = self.model(input_ids)
        elif self.model_type == 'subcategory':
            _, cat_probs = self.model(input_ids)
        cat_loss = self.category_loss_fn(cat_probs, y_cat)
        correct = (cat_probs.argmax(dim=1) == y_cat).sum().item()
        cat_loss.backward()
        self.optimizer.step()
        return cat_loss.item(), correct

    def validate(self):
        self.model.eval()
        total_val_loss, correct_val = 0, 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids, y_cat = [item.to(self.device) for item in batch[:2]]
                if self.model_type == 'category':
                    cat_probs, _ = self.model(input_ids)
                elif self.model_type == 'subcategory':
                    _, cat_probs = self.model(input_ids)
                cat_loss = self.category_loss_fn(cat_probs, y_cat)
                total_val_loss += cat_loss.item()
                correct_val += (cat_probs.argmax(dim=1) == y_cat).sum().item()
        avg_val_loss = total_val_loss / len(self.val_dataloader)
        val_acc = correct_val / len(self.val_dataloader.dataset)
        return avg_val_loss, val_acc

    def _update_history(self, train_loss, train_acc, val_loss, val_acc):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.scheduler.step(val_loss)

    def _check_early_stopping(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1

    def save_model(self, save_path):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), save_path)

def main():

    learning_rate = 1e-5
    epochs = 2
    batch_size = 64

    data_obj = DataPreprocessor('data/main.csv')
    num_categories = data_obj.num_categories
    num_subcategories = data_obj.num_subcategories
    print(f"Number of categories: {num_categories}")
    print(f"Number of subcategories: {num_subcategories}")

    cat_train_dataloader, cat_val_dataloader, sub_train_dataloader, sub_val_dataloader = data_obj.prepare_for_training()


    # Model initialization
    model = BertModel(num_categories, num_subcategories)

    # Train and save models
    # category_model = Trainer(cat_model, 'category', cat_train_dataloader, cat_val_dataloader, epochs, learning_rate)
    # category_model.plot_training_history()
    
    sub_category_model = TrainModel(model, 'subcategory', sub_train_dataloader, sub_val_dataloader, epochs, learning_rate)
    history = sub_category_model.train()
    plot_training_history(history)

if __name__ == '__main__':
    main()
