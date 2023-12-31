from PyQt6.QtWidgets import QMessageBox, QWidget
from PyQt6.QtGui import QIcon
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dark_style_path = os.path.join(current_dir, "dark_style.qss")
light_style_path = os.path.join(current_dir, "light_style.qss")

class ThemeManager:
    dark_mode = True  # Class variable as a default value

    def __init__(self):
        pass 

    @classmethod
    def apply_theme(cls, widget):
        stylesheet_path = dark_style_path if cls.dark_mode else light_style_path
        with open(stylesheet_path, "r") as f:
            stylesheet = f.read()
        cls._apply_stylesheet_to_widget_and_children(widget, stylesheet)
        

    @classmethod
    def _apply_stylesheet_to_widget_and_children(cls, widget, stylesheet):
        widget.setStyleSheet(stylesheet)
        widget.update()  # Add this line
        for child in widget.findChildren(QWidget):
            child.setStyleSheet(stylesheet)
            child.update()

    @classmethod
    def update_widget_theme(cls, widget):
        cls.apply_theme(widget)
        widget.update_content()
        widget.update_graph()

    @classmethod
    def get_plotly_bgcolor(cls):
        return '#2c2c2c' if cls.dark_mode else '#e1e1e1'
    
    @classmethod
    def get_webview_css(cls):
        if cls.dark_mode:
            return '''
            <style>
                body {
                    background-color: #2c2c2c;
                    color: #e1e1e1;
                }
                .gtitle {
                    color: red; /* Change this to your desired color */
                    font-family: "Open Sans", verdana, arial, sans-serif;
                    font-size: 17px;
                    font-weight: normal;
                }
            </style>
            '''
        else:
            return '''
            <style>
                body {
                    background-color: #e1e1e1;
                    color: black;
                }
                /* Include additional styles if needed */
            </style>
            '''
class MessageBox(QMessageBox):
    dark_mode = True  # Class variable as a default value

    def __init__(self):
        super().__init__()
        self.theme_manager = ThemeManager()
        self.setWindowIcon(QIcon('utils/icons/sorting_icon.png'))

    def show_critical(self, title, text):
        ThemeManager.apply_theme(self)
        self.setWindowTitle(title)
        self.setInformativeText(text)
        self.setIcon(QMessageBox.Icon.Critical)  
        return self.exec()

    def show_information(self, title, text):
        ThemeManager.apply_theme(self)
        self.setText(title)
        self.setInformativeText(text)
        self.setIcon(QMessageBox.Icon.Information) 
        return self.exec()

    def show_warning(self, title, text):
        ThemeManager.apply_theme(self)
        self.setText(title)
        self.setInformativeText(text)
        self.setIcon(QMessageBox.Icon.Warning)  
        return self.exec()
