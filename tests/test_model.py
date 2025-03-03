import pytest
import pandas as pd
import torch
import torch.nn as nn  # Added for nn.Module
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification  # Added for BertForSequenceClassification
from .model import BertModel, init_model_data, plot_training_history, train_model  # Added train_model
from .data_prep import DataPreprocessor

@pytest.fixture
def sample_df():
    """Fixture providing a small sample DataFrame for testing."""
    data = {
        'Description': ['buying a book', 'gas for car', 'diapers for baby'],
        'Category': ['Entertainment', 'Auto', 'Baby'],
        'Sub_Category': ['Books', 'Gas', 'Diapers']
    }
    return pd.DataFrame(data)

def test_datapreprocessor_init(sample_df):
    """Test DataPreprocessor initialization with a DataFrame."""
    dp = DataPreprocessor(sample_df)
    df = dp.get_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)  # 3 rows, 3 columns
    assert list(df.columns) == ['Description', 'Category', 'Sub_Category']

def test_datapreprocessor_pop_columns(sample_df):
    """Test that pop_columns adds expected columns and reorders them."""
    dp = DataPreprocessor(sample_df)
    dp.pop_columns()
    df = dp.get_df()
    expected_columns = ['Description', 'Category', 'Sub_Category', 'Tok_Cat', 'Tok_Sub', 'Tokenized']
    assert list(df.columns) == expected_columns
    assert df['Tok_Cat'].iloc[0] == ''  # Initially empty

def test_datapreprocessor_clean_dataframe(sample_df):
    """Test that clean_dataframe processes descriptions correctly."""
    dp = DataPreprocessor(sample_df)
    dp.pop_columns()
    dp.clean_dataframe()
    df = dp.get_df()
    # Check cleaning: lowercase, remove stop words ('a', 'for'), etc.
    assert df['Description'].iloc[0] == 'buying book'
    assert df['Description'].iloc[1] == 'gas car'
    assert df['Description'].iloc[2] == 'diapers baby'

def test_datapreprocessor_tokenize_data(sample_df):
    """Test that tokenize_data adds tokenized columns with correct shapes."""
    dp = DataPreprocessor(sample_df)
    dp.pop_columns()
    dp.clean_dataframe()
    dp.tokenize_data(max_len=50)
    df = dp.get_df()
    assert 'Tokenized_padded' in df.columns
    assert len(df['Tokenized_padded'].iloc[0]) == 50  # Matches max_len
    assert isinstance(df['Tokenized_padded'].iloc[0], list)
    assert isinstance(df['Tok_Cat'].iloc[0], np.ndarray)
    assert df['Tok_Cat'].iloc[0].shape == (13,)  # 13 predefined categories

def test_datapreprocessor_prepare_data(sample_df):
    """Test that prepare_data splits data correctly."""
    dp = DataPreprocessor(sample_df)
    dp.pop_columns()
    dp.clean_dataframe()
    dp.tokenize_data()
    X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test = dp.prepare_data()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_cat_train) == len(X_train)
    assert len(y_sub_train) == len(X_train)
    assert len(y_cat_test) == len(X_test)
    assert isinstance(y_cat_train[0], np.ndarray)  # One-hot encoded

def test_get_cat_sub_numbers(sample_df):
    """Test that category and subcategory counts match predefined values."""
    dp = DataPreprocessor(sample_df)
    num_cat, num_sub = dp.get_cat_sub_numbers()
    assert num_cat == 13  # Number of keys in categories dict
    assert num_sub > 0    # Exact number depends on unique subcategories

def test_bertmodel_init():
    """Test BertModel initialization with correct number of labels."""
    num_categories = 13
    num_subcategories = 50  # Approximate, based on unique subcategories
    model = BertModel(num_categories, num_subcategories)
    assert model.num_categories == num_categories
    assert model.num_subcategories == num_subcategories
    assert isinstance(model.bert_model, BertForSequenceClassification)

def test_bertmodel_forward():
    """Test that the forward pass outputs logits of correct shapes."""
    num_categories = 13
    num_subcategories = 50
    model = BertModel(num_categories, num_subcategories)
    input_ids = torch.randint(0, 1000, (2, 50))  # Batch size 2, seq len 50
    cat_logits, sub_logits = model(input_ids)
    assert cat_logits.shape == (2, num_categories)
    assert sub_logits.shape == (2, num_subcategories)

def test_init_model_data(monkeypatch, sample_df):
    """Test that init_model_data returns correct objects and shapes."""
    class MockDataPreprocessor:
        def __init__(self, data_input):
            self.df = sample_df
        def get_cat_sub_numbers(self):
            return 13, 50
        def pop_columns(self):
            return self.df
        def clean_dataframe(self):
            return self.df
        def tokenize_data(self):
            self.df['Tokenized_padded'] = [[1]*50]*3
            self.df['Tok_Cat'] = [np.ones(13)]*3
            self.df['Tok_Sub'] = [np.ones(50)]*3
            return self.df
        def prepare_data(self):
            X = [[1]*50]*3
            y_cat = [np.ones(13)]*3
            y_sub = [np.ones(50)]*3
            return X[:2], X[2:], y_cat[:2], y_cat[2:], y_sub[:2], y_sub[2:]
        def get_df(self):
            return self.df

    # Updated module path
    monkeypatch.setattr("utils.model.DataPreprocessor", MockDataPreprocessor)

    cat_model, sub_model, cat_train_dl, cat_val_dl, sub_train_dl, sub_val_dl, device, num_cat, num_sub = init_model_data()
    
    assert isinstance(cat_model, BertModel)
    assert isinstance(sub_model, BertModel)
    assert isinstance(cat_train_dl, DataLoader)
    assert isinstance(device, torch.device)
    assert num_cat == 13
    assert num_sub == 50

    for batch in cat_train_dl:
        input_ids, y_cat = batch
        assert input_ids.shape[1] == 50  # Sequence length
        assert y_cat.shape[0] == input_ids.shape[0]
        break

def test_plot_training_history():
    """Test that plot_training_history runs without errors."""
    history = {
        'train_loss': [0.5, 0.4],
        'train_acc': [0.8, 0.85],
        'val_loss': [0.6, 0.5],
        'val_acc': [0.75, 0.8]
    }
    try:
        plot_training_history(history)
    except Exception as e:
        pytest.fail(f"plot_training_history raised an exception: {e}")

def test_plot_training_history_missing_key():
    """Test that plot_training_history handles missing keys."""
    history = {'train_loss': [0.5], 'train_acc': [0.8]}  # Missing val_loss, val_acc
    # Since it prints an error and returns, just ensure it doesn’t crash
    plot_training_history(history)  # Should not raise an exception

def test_train_model_mock(monkeypatch):
    """Basic test for train_model with mock data."""
    class MockModel(nn.Module):
        def forward(self, input_ids):
            return torch.randn(2, 13), torch.randn(2, 50)

    train_ds = TensorDataset(torch.ones(4, 50), torch.zeros(4, dtype=torch.long))
    val_ds = TensorDataset(torch.ones(2, 50), torch.zeros(2, dtype=torch.long))
    train_dl = DataLoader(train_ds, batch_size=2)
    val_dl = DataLoader(val_ds, batch_size=2)

    model = MockModel()
    history = train_model(model, 'category', train_dl, val_dl, epochs=1, learning_rate=1e-5, device='cpu')
    assert 'train_loss' in history
    assert len(history['train_loss']) == 1