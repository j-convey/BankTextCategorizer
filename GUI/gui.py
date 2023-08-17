import sys
import csv
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QListWidget, QFileDialog, QVBoxLayout, QWidget, QGridLayout, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QLabel, QLineEdit, QPushButton, QListWidget, QWidget
from PyQt5.QtGui import QIcon, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QUrl, QStringListModel
from utils import DataPreprocessor
from transformers import BertTokenizer
import pandas as pd
from keras.models import load_model
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.model import BertModel
from utils.dicts import categories
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class LogicHandler:
    def __init__(self):
        self.combined_data = None
        self.category_keys = list(categories.keys())
        self.category_values = [item for sublist in categories.values() for item in sublist]
        self.num_categories = len(self.category_keys)
        self.num_subcategories = len(self.category_values)
        self.model = BertModel(self.num_categories, self.num_subcategories)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder_cat = LabelEncoder()
        self.label_encoder_subcat = LabelEncoder()
        self.onehot_encoder_cat = OneHotEncoder(sparse_output=False)
        self.onehot_encoder_subcat = OneHotEncoder(sparse_output=False)
        self.integer_encoded_cat = self.label_encoder_cat.fit_transform(self.category_keys)
        self.onehot_encoded_cat = self.onehot_encoder_cat.fit_transform(self.integer_encoded_cat.reshape(-1, 1))
        # Encode category_values using label_encoder_subcat
        self.integer_encoded_subcat = self.label_encoder_subcat.fit_transform(self.category_values)
        self.onehot_encoded_subcat = self.onehot_encoder_subcat.fit_transform(self.integer_encoded_subcat.reshape(-1, 1))
        # Create dictionaries for category and sub-category mapping
        self.category_mapping = dict(zip(self.category_keys, self.onehot_encoded_cat))
        self.subcategory_mapping = dict(zip(self.category_values, self.onehot_encoded_subcat))
        self.num_categories = len(self.category_keys)
        # Number of subcategory
        self.num_subcategories = len(self.subcategory_mapping.keys())

    def load_model(self, model_path):
        # Load saved model weights
        state_dict = torch.load(model_path)
        # Load the model state dictionary into the model architecture
        self.model.load_state_dict(state_dict)
        # Set the model to evaluation mode
        self.model.eval()
        return self.model

    def merge_csv_files(self, csv_files):
        if len(csv_files) < 2 or len(csv_files) > 8:
            raise ValueError("Select between 2 to 8 CSV files for merging.")
        df = pd.read_csv(csv_files[0])
        df = df[['Description', 'Category', 'Sub_Category']]
        for file in csv_files[1:]:
            temp_df = pd.read_csv(file, header=0)[['Description', 'Category', 'Sub_Category']]
            df = pd.concat([df, temp_df], ignore_index=True)
        df.dropna(subset=['Category', 'Sub_Category'], inplace=True)
        self.combined_data = df
        return df
    
    def prep_df(self):
        df = self.combined_data
        df.clean_dataframe()
        X_predict = df.tokenize_predict_data()
        predict_input_ids = torch.tensor(X_predict, dtype=torch.long)
        predict_dataset = TensorDataset(predict_input_ids)
        predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
        print("Length of predict_dataloader:", len(predict_dataloader))
        return predict_dataloader

    def predict(self):
        predict_dataloader = self.prep_df()
        df = self.combined_data
        with open(df, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Description', 'Category', 'Subcategory']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            for batch in predict_dataloader:
                input_ids = batch[0].to(self.device)
                with torch.no_grad():
                    category_probs, subcategory_probs = self.model(input_ids)
                    category_predictions = category_probs.argmax(dim=-1)
                    subcategory_predictions = subcategory_probs.argmax(dim=-1)
                for i in range(input_ids.size(0)):
                    category_name = self.label_encoder_cat.inverse_transform([category_predictions[i].item()])[0]
                    subcategory_name = self.label_encoder_subcat.inverse_transform([subcategory_predictions[i].item()])[0]
                    # Unmap input_ids to the original description
                    single_input_ids = input_ids[i].to('cpu')
                    tokens = self.tokenizer.convert_ids_to_tokens(single_input_ids)
                    description = self.tokenizer.convert_tokens_to_string(tokens).strip()
                    writer.writerow({'Description': description, 'Category': category_name, 'Subcategory': subcategory_name})
        return None



class Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dark_mode = True
        self.logic = LogicHandler()
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()
        self.setFixedSize(600, 300)
        self.setWindowIcon(QIcon('GUI/icons/sorting_icon.png'))
        if self.dark_mode:
            bg_color = "#2c2c2c"
            fg_color = "#e1e1e1"
            btn_color = "#4a4a4a"
            btn_fg_color = "#e1e1e1"
            switch_theme_text = "Switch to Light Mode"
        else:
            bg_color = "#e1e1e1"
            fg_color = "#2c2c2c"
            btn_color = "#a7a7a7"
            btn_fg_color = "#2c2c2c"
            switch_theme_text = "Switch to Dark Mode"
        # Enabling Drag and Drop for the main window
        self.setAcceptDrops(True)
        folder_icon = QIcon('GUI/icons/folder_icon.png')

        # Model Path
        self.model_label = QLabel("Model Path:", self)
        self.model_label.setStyleSheet(f"color: {fg_color}")
        layout.addWidget(self.model_label, 0, 0)

        self.model_entry = QLineEdit(self)
        self.model_entry.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.model_entry, 0, 1)

        self.model_button = QPushButton("Browse", self)
        # Adding icon to model_button after its definition
        
        self.model_button.setIcon(folder_icon)
        self.model_button.clicked.connect(self.load_model_path)
        self.model_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.model_button, 0, 2)

        # CSV Files
        self.csv_label = QLabel("CSV Files:", self)
        self.csv_label.setStyleSheet(f"color: {fg_color}")
        layout.addWidget(self.csv_label, 1, 0)

        self.csv_listbox = QListWidget(self)
        self.csv_listbox.setFixedHeight(100)
        self.csv_listbox.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.csv_listbox, 1, 1)

        self.csv_button = QPushButton("Add CSV", self)
        self.csv_button.setIcon(folder_icon)
        self.csv_button.clicked.connect(self.add_csv_path)
        self.csv_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.csv_button, 1, 2)

        # Merge CSVs
        self.merge_button = QPushButton("Merge CSVs", self)
        self.merge_button.clicked.connect(self.merge_csvs)
        self.merge_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.merge_button, 2, 1)

        # Process
        self.process_button = QPushButton("Process", self)
        # Assuming process_data function exists
        # self.process_button.clicked.connect(self.process_data)
        self.process_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.process_button, 3, 1)

        # Switch Theme Button
        self.switch_theme_button = QPushButton(switch_theme_text, self)
        self.switch_theme_button.clicked.connect(self.switch_theme)
        self.switch_theme_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.switch_theme_button, 4, 1)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.setWindowTitle("Bank Description Categorizer")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(f"background-color: {bg_color}")

    def switch_theme(self):
        self.dark_mode = not self.dark_mode
        self.init_ui()

    def dragEnterEvent(self, event: QDragEnterEvent):
        # Checking if the dragged object contains URLs (i.e., file paths)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        # Extracting file paths from the dropped object
        file_urls = event.mimeData().urls()
        
        for url in file_urls:
            file_path = url.toLocalFile()

            # Checking if the file is a CSV before adding it to the list
            if file_path.endswith('.csv'):
                self.csv_files.append(file_path)
                self.csv_listbox.addItem(file_path)

    def load_model_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Model File")
        if file_path:
            self.model_entry.setText(file_path)

    def add_csv_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV files (*.csv)")
        if file_path:
            self.csv_files.append(file_path)
            self.csv_listbox.addItem(file_path)

    def merge_csvs(self):
        try:
            merged_df = self.logic.merge_csv_files(self.csv_files)
            QMessageBox.information(self, "Success", "CSV files merged successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def merge_csv_files(self, *csv_files):
        # Check if the number of csv files is between 2 and 8
        if len(csv_files) < 2 or len(csv_files) > 8:
            print("Error: Invalid number of CSV files.")
            return None
        # Read the first csv file and create the dataframe
        df = pd.read_csv(csv_files[0])
        # Remove all columns except 'Description', 'Category', and 'Sub_Category'
        df = df[['Description', 'Category', 'Sub_Category']]
        # Combine remaining csv files to the dataframe
        for file in csv_files[1:]:
            temp_df = pd.read_csv(file, header=0)[['Description', 'Category', 'Sub_Category']]
            df = pd.concat([df, temp_df], ignore_index=True)
        df.dropna(subset=['Category', 'Sub_Category'], inplace=True)
        print("Successfully merged all CSV files into one dataframe.")
        return df
    
    def process_data(self):
        model_path = self.model_entry.get()
        csv_file = self.csv_entry.get()
        if not model_path or not csv_file:
            QMessageBox.showerror("Error", "Please select the model and CSV file before processing.")
            return
        try:
            # Load the model
            loaded_model = self.logic.load_model(model_path)
            # Predict using the model
            self.logic.predict(csv_file)
            QMessageBox.showinfo("Success", "Data processed successfully!")
        except Exception as e:
            QMessageBox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = Application()
    main_window.show()
    sys.exit(app.exec_())
