import matplotlib.pyplot as plt

def plot_training_history(data):
    expected_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
    for key in expected_keys:
        if key not in data.history.keys():
            print(f"Error: Expected key {key} not found in history")
            return
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(data.history['train_loss'], label='Training Loss')
    plt.plot(data.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(data.history['train_acc'], label='Training Accuracy')
    plt.plot(data.history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()