import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from .data_prep import DataPreprocessor

@pytest.fixture
def mock_dataframe():
    return pd.DataFrame({
        'Description': ['Test description 1', 'Test description 2'],
        'Category': ['Food', 'Auto'],
        'Sub_Category': ['Groceries', 'Gas']
    })

@pytest.fixture
def mock_read_csv():
    mock_df = pd.DataFrame({
        'Description': ['Test description 1', 'Test description 2'],
        'Category': ['Food', 'Auto'],
        'Sub_Category': ['Groceries', 'Gas']
    })
    with patch('pandas.read_csv', return_value=mock_df) as mock_csv:
        yield mock_csv

@pytest.fixture
def mock_bert_tokenizer():
    mock_tokenizer = Mock()
    mock_tokenizer.tokenize.return_value = ['test', 'token']
    mock_tokenizer.convert_tokens_to_ids.return_value = [101, 202]
    with patch('transformers.BertTokenizer.from_pretrained', return_value=mock_tokenizer) as mock_pretrained:
        yield mock_tokenizer

@pytest.fixture
def mock_random():
    with patch('random.sample', side_effect=lambda x, y: x) as mock_sample:
        yield mock_sample

@pytest.fixture
def mock_train_test_split():
    with patch('sklearn.model_selection.train_test_split', return_value=(
        [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]
    )) as mock_split:
        yield mock_split

def test_init_with_file_path(mock_read_csv):
    dp = DataPreprocessor('data/main.csv')
    mock_read_csv.assert_called_once_with('data/main.csv')
    assert isinstance(dp.df, pd.DataFrame)
    assert dp.file_name == 'main'

def test_init_with_dataframe(mock_dataframe):
    dp = DataPreprocessor(mock_dataframe)
    assert isinstance(dp.df, pd.DataFrame)
    assert dp.file_name == ""  
    assert dp.df.equals(mock_dataframe)

def test_init_invalid_input():
    with pytest.raises(ValueError, match="data_input must be a file path or a pandas DataFrame"):
        DataPreprocessor(123)

# Test get_df
def test_get_df(mock_dataframe):
    dp = DataPreprocessor(mock_dataframe)
    result = dp.get_df()
    assert isinstance(result, pd.DataFrame)
    assert result.equals(mock_dataframe)

# Test get_cat_sub_numbers
def test_get_cat_sub_numbers(mock_dataframe):
    dp = DataPreprocessor(mock_dataframe)
    num_categories, num_subcategories = dp.get_cat_sub_numbers()
    assert isinstance(num_categories, int)
    assert isinstance(num_subcategories, int)
    assert num_categories == 13 
    assert num_subcategories == 58 

# Test tokenize_data
def test_tokenize_data(mock_dataframe, mock_bert_tokenizer):
    dp = DataPreprocessor(mock_dataframe)
    result = dp.tokenize_data(max_len=50)
    assert isinstance(result, pd.DataFrame)
    assert 'Tokenized' in result.columns
    assert 'Tokenized_ids' in result.columns
    assert 'Tokenized_padded' in result.columns
    mock_bert_tokenizer.tokenize.assert_called()
    mock_bert_tokenizer.convert_tokens_to_ids.assert_called()

# Test prepare_data
def test_prepare_data(mock_dataframe, mock_train_test_split):
    dp = DataPreprocessor(mock_dataframe)
    dp.tokenize_data()  # Ensure tokenized data exists
    X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test = dp.prepare_data()
    assert isinstance(X_train, list)
    assert isinstance(X_test, list)
    assert isinstance(y_cat_train, list)
    assert isinstance(y_cat_test, list)
    assert isinstance(y_sub_train, list)
    assert isinstance(y_sub_test, list)
    assert len(X_train) > 0
    assert len(X_test) > 0

# Test pop_columns
def test_pop_columns(mock_dataframe):
    dp = DataPreprocessor(mock_dataframe)
    result = dp.pop_columns()
    assert isinstance(result, pd.DataFrame)
    assert 'Description' in result.columns
    assert 'Category' in result.columns
    assert 'Sub_Category' in result.columns
    assert 'Tok_Cat' in result.columns
    assert 'Tok_Sub' in result.columns
    assert 'Tokenized' in result.columns
    assert all(col in result.columns for col in ['Description', 'Category', 'Sub_Category', 'Tok_Cat', 'Tok_Sub', 'Tokenized'])

# Test clean_dataframe
def test_clean_dataframe(mock_dataframe):
    dp = DataPreprocessor(mock_dataframe)
    result = dp.clean_dataframe()
    assert isinstance(result, pd.DataFrame)
    assert 'Description' in result.columns
    assert result['Description'].iloc[0] == 'test description' 
    assert result['Description'].str.lower().equals(result['Description'])  
    assert not result.duplicated(subset=['Description', 'Category']).any() 

# Test clean_predict_data
def test_clean_predict_data(mock_dataframe):
    dp = DataPreprocessor(mock_dataframe)
    result = dp.clean_predict_data()
    assert isinstance(result, pd.DataFrame)
    assert 'Description' in result.columns
    assert result['Description'].iloc[0] == 'test description'
    assert not result.isnull().any().any() 

# Test tokenize_predict_data
def test_tokenize_predict_data(mock_dataframe, mock_bert_tokenizer):
    dp = DataPreprocessor(mock_dataframe)
    dp.clean_predict_data()  # Ensure cleaned data exists
    result = dp.tokenize_predict_data(max_len=105)
    assert isinstance(result, list)
    assert len(result) == len(mock_dataframe)
    assert all(len(seq) <= 105 for seq in result)
    mock_bert_tokenizer.tokenize.assert_called()
    mock_bert_tokenizer.convert_tokens_to_ids.assert_called()

# Test predict_prepare_data
def test_predict_prepare_data(mock_dataframe, mock_bert_tokenizer):
    dp = DataPreprocessor(mock_dataframe)
    dp.clean_predict_data()
    dp.tokenize_predict_data()  # Ensure tokenized data exists
    result = dp.predict_prepare_data()
    assert isinstance(result, list)
    assert len(result) == len(mock_dataframe)

# Test shuffle_sentences
def test_shuffle_sentences(mock_dataframe, mock_random):
    dp = DataPreprocessor(mock_dataframe)
    result = dp.shuffle_sentences()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(mock_dataframe) * 2  # Doubled dataset
    assert 'Description' in result.columns
    assert not result.equals(mock_dataframe) 
    mock_random.assert_called()

if __name__ == '__main__':
    pytest.main(['-v', __file__])