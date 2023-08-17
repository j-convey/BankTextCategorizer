import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QListWidget, QFileDialog, QVBoxLayout, QWidget, QGridLayout, QMessageBox
from PyQt5.QtCore import Qt
import pandas as pd

class Application(QMainWindow):

    def __init__(self):
        super().__init__()

        self.dark_mode = True  # Default mode set to dark
        self.csv_files = []

        self.init_ui()

    def init_ui(self):

        layout = QGridLayout()

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

        # Model Path
        self.model_label = QLabel("Model Path:", self)
        self.model_label.setStyleSheet(f"color: {fg_color}")
        layout.addWidget(self.model_label, 0, 0)

        self.model_entry = QLineEdit(self)
        self.model_entry.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.model_entry, 0, 1)

        self.model_button = QPushButton("Browse", self)
        self.model_button.clicked.connect(self.load_model_path)
        self.model_button.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.model_button, 0, 2)

        # CSV Files
        self.csv_label = QLabel("CSV Files:", self)
        self.csv_label.setStyleSheet(f"color: {fg_color}")
        layout.addWidget(self.csv_label, 1, 0)

        self.csv_listbox = QListWidget(self)
        self.csv_listbox.setFixedHeight(150)
        self.csv_listbox.setStyleSheet(f"background-color: {btn_color}; color: {btn_fg_color}")
        layout.addWidget(self.csv_listbox, 1, 1)

        self.csv_button = QPushButton("Add CSV", self)
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
        if len(self.csv_files) < 2 or len(self.csv_files) > 8:
            QMessageBox.critical(self, "Error", "Select between 2 to 8 CSV files for merging.")
            return

        try:
            merged_df = self.merge_csv_files(*self.csv_files)
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
    
    '''def process_data(self):
        model_path = self.model_entry.get()
        csv_file = self.csv_entry.get()

        if not model_path or not csv_file:
            messagebox.showerror("Error", "Please select the model and CSV file before processing.")
            return

        try:
            # Load the model
            loaded_model = self.load_model(model_path)

            # Predict using the model
            self.predict(loaded_model, csv_file)

            # Show success message
            messagebox.showinfo("Success", "Data processed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")'''

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = Application()
    main_window.show()
    sys.exit(app.exec_())
