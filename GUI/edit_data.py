from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton
from PyQt6.QtCore import Qt, pyqtSlot
import pandas as pd
from graph_logic import GraphLogic
class EditData(QWidget):
    def __init__(self):
        super(EditData, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.graphlogic = GraphLogic()
        df = self.graphlogic.return_df()  # Assuming this returns a pandas DataFrame

        self.layout = QVBoxLayout()

        self.tableWidget = QTableWidget()

        # Set table size to DataFrame size + 5 additional rows for new data
        self.tableWidget.setRowCount(df.shape[0] + 5)
        self.tableWidget.setColumnCount(df.shape[1])

        # Set DataFrame columns as table header labels
        self.tableWidget.setHorizontalHeaderLabels(df.columns.tolist())

        # Make the "Description" column twice as wide as the first column (change this as necessary)
        self.tableWidget.setColumnWidth(0, 100)  # Assuming the first column width is 100
        description_col_index = df.columns.get_loc("Description")
        self.tableWidget.setColumnWidth(description_col_index, 200)  # Double the width

        # Fill the table with DataFrame values
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                cell_value = str(df.iloc[i, j])
                self.tableWidget.setItem(i, j, QTableWidgetItem(cell_value))

        # Add the table to the layout
        self.layout.addWidget(self.tableWidget)

        # Connect the cellChanged signal to a custom slot
        self.tableWidget.cellChanged.connect(self.cell_data_changed)

        # Set the layout
        self.setLayout(self.layout)

    @pyqtSlot(int, int)
    def cell_data_changed(self, row, column):
        # Check if the edited cell is in the last 5 rows
        if row >= self.tableWidget.rowCount() - 5:
            # Add an additional row
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
