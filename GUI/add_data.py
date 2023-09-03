from PyQt6.QtWidgets import (QApplication, QMainWindow, QGridLayout, QLabel, QLineEdit, QListWidget, QPushButton, 
                             QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, 
                             QDialog, QProgressDialog, QSizePolicy, QAbstractItemView, QHBoxLayout, QTabWidget,)
from PyQt6.QtGui import QIcon, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QProgressDialog
from PyQt6.QtCore import QThreadPool, QRunnable, pyqtSlot, QObject, pyqtSignal, Qt
import pandas as pd
from message_box import MessageBox, ThemeManager
from logic import LogicHandler
from PyQt6.QtWidgets import QAbstractItemView

class WorkerSignals(QObject):
    finished = pyqtSignal()

class Worker(QRunnable):
    def __init__(self, function_to_run, *args, **kwargs):
        super(Worker, self).__init__()
        self.function_to_run = function_to_run
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        self.function_to_run(*self.args, **self.kwargs)
        self.signals.finished.emit() 

class AddData(QWidget):
    folder_icon = None
    trash_icon = None

    def __init__(self):
        super(AddData, self).__init__()
        msg_box = MessageBox()
        self.logic = LogicHandler()
        self.csv_files = []
        main_layout = QVBoxLayout()
        layout = QVBoxLayout()

        # Create a horizontal box layout for the CSV label
        csv_header_layout = QHBoxLayout()
        csv_header_layout.addStretch(1)
        self.csv_label = QLabel("Add CSV Files")
        csv_header_layout.addWidget(self.csv_label)
        csv_header_layout.addStretch(1)

        # Create a horizontal box layout for the Add and Remove buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)
        self.csv_button = QPushButton("Add CSV")
        self.csv_button.setIcon(self.folder_icon)
        self.csv_button.clicked.connect(self.add_csv_path)  # Assuming this is defined elsewhere
        buttons_layout.addWidget(self.csv_button)

        self.remove_csv_button = QPushButton("Remove CSV")
        self.remove_csv_button.setIcon(self.trash_icon)
        self.remove_csv_button.clicked.connect(self.remove_csv_path)  # Assuming this is defined elsewhere
        buttons_layout.addWidget(self.remove_csv_button)
        buttons_layout.addStretch(1)

        # CSV List Box
        self.csv_listbox = QListWidget()
        self.csv_listbox.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.csv_listbox.setAcceptDrops(True)
        self.csv_listbox.setDropIndicatorShown(True)
        self.csv_listbox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Merge and Process buttons
        self.merge_button = QPushButton("Merge CSVs")
        self.merge_button.clicked.connect(self.merge_csvs)  # Assuming this is defined elsewhere
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_data)  # Assuming this is defined elsewhere

        # Populate the layout with widgets
        layout.addLayout(csv_header_layout)
        layout.addWidget(self.csv_listbox)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.merge_button)
        layout.addWidget(self.process_button)

        # Add the non-growing layout to the main layout, centering it
        main_layout.addStretch(1)
        main_layout.addLayout(layout)
        main_layout.addStretch(1)

        self.setLayout(main_layout)

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
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open CSV File", "", "CSV files (*.csv)")
        for file_path in file_paths:
            self.csv_files.append(file_path)
            self.csv_listbox.addItem(file_path)
    
    def remove_csv_path(self):
        msg_box = MessageBox()
        selected_items = self.csv_listbox.selectedItems()
        if not selected_items:  # No item selected
            msg_box.show_critical("Error", "Please select a CSV file to remove.")
        for item in selected_items:
            self.csv_files.remove(item.text())  # Remove from list
            self.csv_listbox.takeItem(self.csv_listbox.row(item))  # Remove from GUI listbox

    def merge_csvs(self):
        msg_box = MessageBox()
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
        data_preview.setFixedWidth(total_width + 60) 
        layout.addWidget(table)
        data_preview.setLayout(layout)
        data_preview.exec_()

    def process_data(self):
        msg_box = MessageBox()
        if len(self.csv_files) == 0:
            msg_box.show_critical("Error", "Please select CSV file before processing.")
            return

        try:
            if len(self.csv_files) == 1:
                df_to_process = pd.read_csv(self.csv_files[0])
            else:
                df_to_process = self.logic.combined_data

            # Initialize QProgressDialog
            progress = QProgressDialog("Processing data...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            self.logic.progress_signal.connect(progress.setValue)
            progress.show()

            # Connect the progress_signal to a lambda function to update QProgressDialog
            self.logic.progress_signal.connect(lambda x: progress.setValue(x))

            # Create a new worker thread and move your logic to that thread
            worker = Worker(self.logic.predict, df_to_process)
            worker.signals.finished.connect(self.on_worker_finished)  # Connect to worker's finished signal
            QThreadPool.globalInstance().start(worker)

        except Exception as e:
            msg_box.show_critical("Error", f"An error occurred: {str(e)}")

    def on_worker_finished(self):
        msg_box = MessageBox()
        try:
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
            msg_box.show_critical("Error", f"An error occurred during saving: {str(e)}")
