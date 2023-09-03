import sys
import csv
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGridLayout, QLabel, QLineEdit, QListWidget, QPushButton, 
                             QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, 
                             QDialog, QProgressDialog, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QUrl, QStringListModel
from data_prep import DataPreprocessor
from model import BertModel
from dicts import categories

class LogicHandler:
    def __init__(self):
        self.cat_model_path = 'models/pt_cat_modelV1'
        self.sub_model_path = 'models/pt_sub_modelV1'
        self.combined_data = None
        self.category_keys = list(categories.keys())
        # Blank df for writing to CSV
        self.predict_df = pd.DataFrame()
        self.category_values = [item for sublist in categories.values() for item in sublist]
        self.num_categories = len(self.category_keys)
        self.num_subcategories = len(self.category_values)
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
    def return_processed_data(self):
        return self.predict_df
    
    def view_data(self):
        return self.combined_data
    
    def load_cat_model(self):
        cat_model = BertModel(self.num_categories, self.num_subcategories)
        state_dict = torch.load(self.cat_model_path)
        cat_model.load_state_dict(state_dict)
        # Set the models to evaluation mode
        cat_model.eval()
        return cat_model

    def load_sub_model(self):
        sub_model = BertModel(self.num_categories, self.num_subcategories)
        # Load saved model weights
        state_dict = torch.load(self.sub_model_path)
        sub_model.load_state_dict(state_dict)
        sub_model.eval()
        return sub_model

    def merge_csv_files(self, csv_files):
        if len(csv_files) < 2 or len(csv_files) > 8:
            raise ValueError("Select between 2 to 8 CSV files for merging.")
        df = pd.read_csv(csv_files[0])
        df = df[['Description', 'Category', 'Sub_Category']]
        for file in csv_files[1:]:
            temp_df = pd.read_csv(file, header=0)[['Description', 'Category', 'Sub_Category']]
            df = pd.concat([df, temp_df], ignore_index=True)
        self.combined_data = df
        return df
    
    def prep_df(self, df):
        # print df type
        print(type(df))
        df_obj = DataPreprocessor(df)
        df_obj.clean_dataframe()
        X_predict = df_obj.tokenize_predict_data()
        predict_input_ids = torch.tensor(X_predict, dtype=torch.long)
        predict_dataset = TensorDataset(predict_input_ids)
        predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
        print("Length of predict_dataloader:", len(predict_dataloader))
        return predict_dataloader

    def predict(self, df, progress_callback=None):
        # Load models
        cat_model = self.load_cat_model()
        sub_model = self.load_sub_model()
        cat_model.to(self.device)
        sub_model.to(self.device)
        output_csv = 'predict_data_test.csv'
        predict_dataloader = self.prep_df(df)
        total_items = len(predict_dataloader.dataset)
        # Lists to store predictions and descriptions
        categories, subcategories, descriptions = [], [], []
        for batch in predict_dataloader:
            input_ids = batch[0].to(self.device)
            with torch.no_grad():
                category_probs, _ = cat_model(input_ids)
                category_predictions = category_probs.argmax(dim=-1)
                _, subcategory_probs = sub_model(input_ids)
                subcategory_predictions = subcategory_probs.argmax(dim=-1)
            for i in range(input_ids.size(0)):
                category_name = self.label_encoder_cat.inverse_transform([category_predictions[i].item()])[0]
                categories.append(category_name)
                subcategory_name = self.label_encoder_subcat.inverse_transform([subcategory_predictions[i].item()])[0]
                subcategories.append(subcategory_name)
                # Unmap input_ids to the original description
                single_input_ids = input_ids[i].to('cpu')
                tokens = self.tokenizer.convert_ids_to_tokens(single_input_ids)
                description = self.tokenizer.convert_tokens_to_string([token for token in tokens if token != "[PAD]"]).strip()
                descriptions.append(description)    
        # Write to CSV
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Description', 'Category', 'Subcategory']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            for description, category, subcategory in zip(descriptions, categories, subcategories):
                writer.writerow({'Description': description, 'Category': category, 'Subcategory': subcategory})
        self.predict_df = pd.read_csv(output_csv)
        return output_csv

class StyledMessageBox(QMessageBox):
    def __init__(self, dark_mode=True):
        super().__init__()
        if dark_mode:
            bg_color = "#2c2c2c"
            fg_color = "#e1e1e1"
        else:
            bg_color = "#e1e1e1"
            fg_color = "#2c2c2c"
        self.setStyleSheet(f"QMessageBox {{background-color: {bg_color}; color: {fg_color};}}")
        self.setWindowIcon(QIcon('utils/icons/sorting_icon.png'))
        
    def show_critical(self, title, text):
        self.setWindowTitle(title)
        self.setInformativeText(text)
        self.setIcon(QMessageBox.Critical)
        return self.exec_()

    def show_information(self, title, text):
        self.setText(title)
        self.setInformativeText(text)
        self.setIcon(QMessageBox.Information)
        return self.exec_()

    def show_warning(self, title, text):
        self.setText(title)
        self.setInformativeText(text)
        self.setIcon(QMessageBox.Warning)
        return self.exec_()

class Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dark_mode = True
        self.theme_icon_light = QIcon('utils/icons/dark_mode.png')  
        self.theme_icon_dark = QIcon('utils/icons/light_mode.png')
        self.trash_icon = QIcon('utils/icons/trash.png')
        self.logic = LogicHandler()
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()
        self.setFixedSize(800, 500)
        self.setWindowIcon(QIcon('utils/icons/sorting_icon.png'))
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
        # Enabling Drag and Drop for the main window
        self.setAcceptDrops(True)
        folder_icon = QIcon('utils/icons/folder_icon.png')
        blankWidget = QWidget()
        blankWidget.setFixedSize(25, 25)

        # ROW, COLUMN, ROW SPAN, COLUMN SPAN, ALIGNMENT
        '''Model Version Display'''
        self.model_version_label = QLabel("Model Version: pt_v1", self)
        self.model_version_label.setStyleSheet(f"color: {fg_color}; font-weight: bold;")
        layout.addWidget(self.model_version_label, 0, 4, 1, 3, Qt.AlignBottom)

        '''Switch Theme Button'''
        self.switch_theme_button = QPushButton(self)
        self.switch_theme_button.setIcon(self.theme_icon_dark if self.dark_mode else self.theme_icon_light)  # set initial icon based on mode
        self.switch_theme_button.clicked.connect(self.switch_theme)
        self.switch_theme_button.setFixedSize(32, 32)  # Adjust size to fit icon
        layout.addWidget(self.switch_theme_button, 0, 10, 1, 1, Qt.AlignRight)

        '''CSV Files Button'''
        self.csv_files = []
        self.csv_label = QLabel("CSV Files:", self)
        self.csv_label.setStyleSheet(f"color: {fg_color}")
        layout.addWidget(self.csv_label, 3, 1, 1, 1)

        '''CSV Listbox'''
        self.csv_listbox = QListWidget(self)
        self.csv_listbox.setFixedHeight(200)
        #self.csv_listbox.setFixedWidth(581)
        self.csv_listbox.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.csv_listbox, 2, 2, 3, 8, Qt.AlignBottom)

        '''Add CSV Button'''
        self.csv_button = QPushButton("Add CSV", self)
        self.csv_button.setIcon(folder_icon)
        self.csv_button.clicked.connect(self.add_csv_path)
        self.csv_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.csv_button, 5, 3, 1, 2, Qt.AlignCenter)

        '''CSV Remove Button'''
        self.remove_csv_button = QPushButton("Remove CSV", self)
        self.remove_csv_button.setIcon(self.trash_icon)
        self.remove_csv_button.clicked.connect(self.remove_csv_path)
        self.remove_csv_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.remove_csv_button, 5, 7, 1, 2, Qt.AlignCenter)

        '''Merge CSVs Button'''
        self.merge_button = QPushButton("Merge CSVs", self)
        self.merge_button.clicked.connect(self.merge_csvs)
        self.merge_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.merge_button, 7, 3, 1, 6) 

        '''Process Button'''
        self.process_button = QPushButton("Process", self)
        self.process_button.clicked.connect(self.process_data)  # Connecting the process_data function
        self.process_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.process_button, 8, 3, 1, 6)
        
        layout.addWidget(blankWidget, 9, 0, 1, 12)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.setWindowTitle("Bank Description Categorizer")
        #self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(f"background-color: {bg_color}")

    def apply_theme(self):
        if self.dark_mode:
            bg_color = "#2c2c2c"
            fg_color = "#e1e1e1"
            btn_color = "#4a4a4a"
            btn_fg_color = "#e1e1e1"
            self.switch_theme_button.setIcon(self.theme_icon_dark)
        else:
            bg_color = "#e1e1e1"
            fg_color = "#2c2c2c"
            btn_color = "#a7a7a7"
            btn_fg_color = "#2c2c2c"
            self.switch_theme_button.setIcon(self.theme_icon_light)
        # Applying styles
        self.setStyleSheet(f"background-color: {bg_color}")
        self.model_version_label.setStyleSheet(f"color: {fg_color}; font-weight: bold;")
        self.csv_label.setStyleSheet(f"color: {fg_color}")
        self.csv_listbox.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        self.csv_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        self.remove_csv_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        self.merge_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        self.process_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")

    def switch_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
    
    def apply_widget_theme(self, widget):
        if self.dark_mode:
            bg_color = "#2c2c2c"
            fg_color = "#e1e1e1"
        else:
            bg_color = "#e1e1e1"
            fg_color = "#2c2c2c"
        widget.setStyleSheet(f"background-color: {bg_color}; color: {fg_color};")

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

    def add_csv_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV files (*.csv)")
        if file_path:
            self.csv_files.append(file_path)
            self.csv_listbox.addItem(file_path)
    
    def remove_csv_path(self):
        msg_box = StyledMessageBox(self.dark_mode)
        self.apply_widget_theme(msg_box)
        selected_items = self.csv_listbox.selectedItems()
        if not selected_items:  # No item selected
            msg_box = StyledMessageBox(self.dark_mode)
            self.apply_widget_theme(msg_box)
            msg_box.show_critical("Error", "Please select a CSV file to remove.")
        for item in selected_items:
            self.csv_files.remove(item.text())  # Remove from list
            self.csv_listbox.takeItem(self.csv_listbox.row(item))  # Remove from GUI listbox

    def merge_csvs(self):
        msg_box = StyledMessageBox(self.dark_mode)
        self.apply_widget_theme(msg_box)
        # Display error if no CSV files are selected or if only 1
        if len(self.csv_files) == 0:
            msg_box.show_critical("Error", "Please select at least 2 CSV files to merge.")
            return
        elif len(self.csv_files) == 1:
            msg_box.show_critical("Error", "Please select at least 2 CSV files to merge or click process to process 1 CSV file.")
            return
        try:
            merged_df = self.logic.merge_csv_files(self.csv_files)
            msg_box.show_information("Success", "CSV files merged successfully!")
            self.show_merged_data(merged_df)
        except Exception as e:
            msg_box.critical(self, "Error", f"An error occurred: {str(e)}")

    def show_merged_data(self, df):
        data_preview = QDialog(self)
        self.apply_widget_theme(data_preview)
        data_preview.setWindowTitle("Merged CSV Data Preview")
        layout = QVBoxLayout()
        table = QTableWidget()
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns)
        table.setRowCount(len(df))
        print(df.head()) # Debugging
        for row in range(len(df)):
            for col, col_name in enumerate(df.columns):
                table.setItem(row, col, QTableWidgetItem(str(df.iloc[row][col_name])))
        # Resize columns based on their content
        # After resizing columns and before displaying the QDialog
        total_width = table.verticalHeader().width() + 40  # width of vertical header + a little margin
        for col in range(table.columnCount()):
            total_width += table.columnWidth(col)
        data_preview.setFixedWidth(total_width + 60)  # Additional 40 pixels for some margin/padding
        # Apply custom style for dark mode
        if self.dark_mode:
            table.setStyleSheet("""
                QTableWidget {
                    gridline-color: #5A5A5A;
                    border: none;
                }
                QHeaderView::section {
                    background-color: #3A3A3A;
                    color: #e1e1e1;
                    border: 1px solid #6A6A6A;
                    padding: 5px;
                }
            """)
        layout.addWidget(table)
        data_preview.setLayout(layout)
        data_preview.exec_()
    
    def init_progress_dialog(self):
        self.progress = QMessageBox(self)
        self.progress.setIcon(QMessageBox.Information)
        self.progress.setText("Processing data...")
        self.progress.setWindowTitle("Progress")
        self.progress.setStandardButtons(QMessageBox.NoButton)  # Hide all buttons to act as an indicator
        self.progress.show()

    def process_data(self):
        msg_box = StyledMessageBox(self.dark_mode)
        self.apply_widget_theme(msg_box)
        if len(self.csv_files) == 0:
            msg_box.show_critical("Error", "Please select CSV file before processing.")
            return
        elif len(self.csv_files) == 1:
            df_to_process = pd.read_csv(self.csv_files[0])
        elif self.logic.combined_data is None:
            msg_box.show_critical("Error", "Please merge CSV files before processing.")
            return
        else:
            df_to_process = self.logic.combined_data
        try:
            self.logic.predict(df_to_process)
            # Save file dialog
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Processed Data", "", "CSV files (*.csv)")
            if save_path:
                processed_data = self.logic.return_processed_data()
                processed_data.to_csv(save_path, index=False)
                msg_box.show_information("Success", f"Data processed and saved to {save_path}")
                # Show the processed data
                self.show_merged_data(processed_data)
            else:
                msg_box.show_information("Information", "Data processed but not saved.")
        except Exception as e:
            msg_box.show_critical("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = Application()
    main_window.show()
    sys.exit(app.exec_())


