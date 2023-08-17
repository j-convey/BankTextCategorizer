# Bank Transaction Categorization
A simple and powerful tool that uses a neural network to categorize bank transaction descriptions into respective categories and subcategories.

## Overview
This program is designed to help with the categorization of bank transaction descriptions. It leverages the power of the BERT model to recognize patterns in transaction descriptions and subsequently classify them into predefined categories and subcategories.

## Features
- Data preprocessing to clean and prepare the transaction descriptions for the neural network.
- Utilizes the BERT model, a state-of-the-art NLP model, for accurate categorization.
- Simultaneous categorization into main category and subcategory.
- Easy to use interface with simple Python functions.

## How to Use
### Setup & Installation
Make sure you have all the necessary libraries installed. Most importantly, ensure that torch, transformers, and pandas are available.
```bash
pip install torch transformers pandas
```
### Categories/Subcategories
This program comes with a standard set of categories and subcategories to categorize your bank transactions. Below is the default categorization schema:
```python
categories = {
    'Auto': ['Gas','Maintenance', 'Upgrades', 'Other_Auto'],
    'Baby': ['Diapers', 'Formula', 'Clothes', 'Toys', 'Other_Baby'],
    'Clothes': ['Clothes', 'Shoes', 'Jewelry', 'Bags_Accessories'],
    'Entertainment': ['Sports_Outdoors', 'Movies_TV', 'DateNights', 'Arts_Crafts', 'Books', 'Games', 'Guns', 'E_Other'],
    'Electronics': ['Accessories', 'Computer', 'TV', 'Camera', 'Phone','Tablet_Watch', 'Gaming', 'Electronics_misc'],
    'Food': ['Groceries', 'FastFood_Restaurants'],
    'Home': ['Maintenance', 'Furniture_Appliances', 'Hygiene', 'Gym',
        'Home_Essentials', 'Kitchen', 'Decor', 'Security', 'Yard_Garden', 'Tools'],
    'Medical': ['Health_Wellness'],
    'Kids': ['K_Toys'],
    'Personal_Care': ['Hair', 'Makeup_Nails', 'Beauty', 'Massage','Vitamins_Supplements', 'PC_Other'],
    'Pets': ['Pet_Food', 'Pet_Toys', 'Pet_Med', 'Pet_Grooming', 'Pet_Other'],
    'Subscriptions_Memberships': ['Entertainment', 'Gym', 'Sub_Other'],
    'Travel': ['Hotels', 'Flights', 'Car_Rental', 'Activities']
}

```
#### Flexibility in Categorization
However, we understand that every user might have specific needs, and the default categories might not fit everyone. You have the flexibility to modify, add, or remove categories and subcategories as per your requirements.

#### How to Customize:
Update the Dictionary: Modify the categories dictionary in the code with your desired categories and subcategories. The key should be the main category, and the values should be a list containing the subcategories.

Update Training Data: It's crucial that once you modify the categories and subcategories, you also need to change the training data. Ensure that the data has labels corresponding to your new categories and subcategories.

Re-Train the Model: With the updated categories and training data, re-run the main() function to train the model on the new data.

By following these steps, you can easily customize the categorization to suit your personal or business needs.


## UI
![image](https://github.com/j-convey/BankTextCategorizer/assets/85854964/9c88533e-23e1-4989-9d95-e01a53518ab5)

## Code Structure
- Data Preprocessing: DataPreprocessor class is responsible for reading the CSV file, cleaning the data, tokenizing the sentences, and preparing the data for training.
- BERT Model: BertModel class initializes the BERT model with the appropriate number of categories and subcategories.
- Training: train_category_model function handles the training loop, including batching, forward and backward passes, and early stopping.

## Future Improvements
- Implementing the subcategory model training function.
- Refining the data preprocessing pipeline.
- Fine-tuning the BERT model for better performance.
- Implementing a gui for ease of use.
