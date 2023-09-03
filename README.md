# Bank Transaction Categorization
A simple and powerful tool that uses a neural network to categorize bank transaction descriptions into respective categories and subcategories.
## 💡 Motivation
My primary motivation for creating this project was to provide an alternative to existing solutions by emphasizing two critical values:

1. Self-hosting: By allowing users to host the solution on their infrastructure, this project offers complete control over data, operations, and customization. Self-hosting eliminates the dependency on third-party services which might shut down, change their terms, or even introduce pricing that might not be favorable to all.
2. Privacy-friendly: In an era where every click, every view, and even every scroll is tracked, privacy has become a scarce commodity. This project is designed with privacy at its core. No unnecessary data collection, no sneaky trackers, and no third-party analytics. Your data stays yours, and that's how it should be.

## 📋 Overview
This program is designed to help with the categorization of bank transaction descriptions. It leverages the power of the BERT model to recognize patterns in transaction descriptions and subsequently classify them into predefined categories and subcategories.

## ✨ Features
- Data preprocessing to clean and prepare the transaction descriptions for the neural network.
- Utilizes the BERT model, a state-of-the-art NLP model, for accurate categorization.
- Simultaneous categorization into main category and subcategory.
- Easy to use interface with simple Python functions.

## 🚀 How to Use
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


## 🖥️ UI
![image](https://github.com/j-convey/BankTextCategorizer/assets/85854964/6564f384-7181-4daa-a014-17c200f72090)
![image](https://github.com/j-convey/BankTextCategorizer/assets/85854964/f05b291a-cc4b-418f-a328-937eb771da5a)
![image](https://github.com/j-convey/BankTextCategorizer/assets/85854964/9b4a0499-37e2-4e70-846d-ea8dd75bfa26)


## 🏗️ Code Structure
- Data Preprocessing: DataPreprocessor class is responsible for reading the CSV file, cleaning the data, tokenizing the sentences, and preparing the data for training.
- BERT Model: BertModel class initializes the BERT model with the appropriate number of categories and subcategories.
- Training: train_category_model function handles the training loop, including batching, forward and backward passes, and early stopping.

## 📈 Performance
Here are my results after using main.csv dataset (62,793 lines of data) for 2 epochs. This took around 10 hours to complete based on my hardware. This is without using data augmentation to double the size due to time restaints.
![cat_modelV1](https://github.com/j-convey/BankTextCategorizer/assets/85854964/f457198d-4de0-4ef2-b7eb-3f30d6c14d58)


## 🔮 Future Improvements
- Adding Visualizers for viewing categorized data to Summary and Details tab.
- Refining the data preprocessing pipeline.
- Fine-tuning the BERT model for better performance.

## 🤝 Contribute
1. Consider contributing to the project.
2. Add a Github Star to the project.
3. Post about the project on X.

