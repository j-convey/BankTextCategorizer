import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
                avg_train_loss = total_train_loss / (i + 1)
                train_acc = (correct_train / ((i + 1) * len(batch))) / 100
                print(f"Epoch {epoch}/{epochs} - Batch {i+1}/{len(train_dataloader)} "
                      f"- Training loss: {avg_train_loss:.4f}, Training Acc: {train_acc:.4f}, "
                      f"Validation loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.4f}")
                model.train()
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
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

def train_and_save_model(model, model_type, train_dataloader, val_dataloader, save_path, epochs, learning_rate, device):
    """Train a model and save it to the specified path."""
    model.to(device)
    history = train_model(model, model_type, train_dataloader, val_dataloader, epochs, learning_rate, device)
    model.to('cpu')
    torch.save(model.state_dict(), save_path)
    return history