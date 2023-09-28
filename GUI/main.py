from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QStatusBar, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QTabBar, QStyle, QStyleOptionTab, QStylePainter, QSpacerItem, QSizePolicy, QMenuBar, QMenu
from PyQt6.QtGui import QIcon, QDragEnterEvent, QDropEvent, QPalette, QColor, QBrush, QPixmap, QFont, QCursor, QWindow, QAction
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QSizePolicy, QLabel

import sys
from summary import SummaryTab
from details import DetailsTab
from add_data import AddData
from logic import LogicHandler
from message_box import ThemeManager
from edit_data import EditData


class BankingApp(QMainWindow):
    themeChanged = pyqtSignal(bool) 
    def __init__(self):
        super().__init__()
        self.theme_manager = ThemeManager()
        AddData.folder_icon = QIcon('GUI/icons/folder_icon.png')
        AddData.trash_icon = QIcon('GUI/icons/trash.png')
        self.logic = LogicHandler()  # Assuming LogicHandler is defined elsewhere
        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(1200, 700)
        self.setWindowTitle("Banking App")
        # Initialize menu bar
            # Initialize menu bar
        menu_bar = QMenuBar()

        # Create File menu and add actions
        file_menu = QMenu("File", self)
        open_action = QAction("Open", self)
        save_action = QAction("Save", self)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

        # Create Settings menu and add actions
        settings_menu = QMenu("Settings", self)
        preferences_action = QAction("Preferences", self)
        
        # Theme submenu
        theme_menu = QMenu("Theme", self)
        light_mode_action = QAction("Light Mode", self)
        dark_mode_action = QAction("Dark Mode", self)
        light_mode_action.triggered.connect(lambda: self.switch_theme(False))
        dark_mode_action.triggered.connect(lambda: self.switch_theme(True))
        theme_menu.addAction(light_mode_action)
        theme_menu.addAction(dark_mode_action)
        settings_menu.addAction(preferences_action)
        settings_menu.addMenu(theme_menu)  # Add Theme submenu to Settings

        # Add menus to the menu bar
        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(settings_menu)
        self.setMenuBar(menu_bar)

        # Initialize tab widget
        self.tab_widget = QTabWidget()
        self.summary_tab = SummaryTab() 
        self.themeChanged.connect(self.summary_tab.update_theme) 
        self.details_tab = DetailsTab() 
        self.csv_tab = AddData() 
        self.edit_data_tab = EditData()

        self.tab_widget.addTab(self.summary_tab, "Summary")
        self.tab_widget.addTab(self.details_tab, "Details")
        self.tab_widget.addTab(self.csv_tab, "Add CSV Files")
        self.tab_widget.addTab(self.edit_data_tab, "Edit Data")

        # Central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)  # Tab widget takes full size
        central_widget.setLayout(layout)
        self.theme_manager.apply_theme(self)

    def switch_theme(self, dark_mode):
        ThemeManager.dark_mode = dark_mode  # Assuming ThemeManager is defined elsewhere
        self.theme_manager.apply_theme(self)
        self.themeChanged.emit(dark_mode)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BankingApp()
    window.show()
    sys.exit(app.exec())
