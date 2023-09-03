from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QSizePolicy, QSpacerItem, QSizePolicy, QVBoxLayout, QGraphicsDropShadowEffect, QHBoxLayout
from PyQt6.QtCore import Qt, QSize, pyqtSlot
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QColor
import datetime
import plotly.graph_objects as go
import plotly.io as pio
from graph_logic import GraphLogic
from message_box import ThemeManager

class SpendingLabel(QWidget):
    def __init__(self, title, value):
        super(SpendingLabel, self).__init__()

        # Initialize the layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create the title and value labels
        title_label = QLabel(title)
        value_label = QLabel(value)

        # Style the labels
        title_label.setStyleSheet("""
            font-size: 14px;
            color: #555;
        """)

        value_label.setStyleSheet("""
            font-size: 20px;
            color: #333;
            font-weight: bold;
        """)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(3, 3)
        self.setGraphicsEffect(shadow)

        # Add the labels to the layout
        layout.addWidget(title_label)
        layout.addWidget(value_label)

        # Style the widget itself
        self.setStyleSheet("""
            background-color: #EEE;
            border-radius: 10px;
            padding: 15px;
        """)

        # Center align the text
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set layout to remove extra space
        layout.setContentsMargins(0, 0, 0, 0)



class SummaryTab(QWidget):
    def __init__(self):
        super(SummaryTab, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.graph_logic = GraphLogic()
        self.web_view = QWebEngineView()
        self.mtd_web_view = QWebEngineView() 
        self.set_up_graph(self.web_view, self.graph_logic.ytd_cat_subcat_sunburst())
        self.set_up_graph(self.mtd_web_view, self.graph_logic.mtd_cat_subcat_sunburst())

        current_month, current_year = self.graph_logic.return_monthyear()
        mtd_spending = self.graph_logic.mtd_spending()
        ytd_spending = self.graph_logic.ytd_spending()

        summary_label = QLabel(f"<b>Summary for {current_year}</b>")
        summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        current_month_label = QLabel(f"Current spending for {datetime.date(current_year, current_month, 1).strftime('%B')}: ${mtd_spending}")
        current_month_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ytd_spending_label = QLabel(f"YTD spending for {current_year}: ${ytd_spending}")
        ytd_spending_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set fixed size policy to prevent labels from expanding
        summary_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        current_month_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        ytd_spending_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        # Create the sunburst plot with a transparent background
        

        # Set a fixed size or maximum size if you want to control the dimensions
        self.web_view.setMaximumSize(QSize(600, 400))
        self.web_view.setMinimumSize(QSize(600, 400))

        # Add the CSS to your existing HTML


        summary_label.setMaximumHeight(40)
        current_month_label.setMaximumHeight(40)
        ytd_spending_label.setMaximumHeight(40)
        #ytd_spending label to not expand
        ytd_spending_label.setProperty("class", "no_expand")
        # Create a QGridLayout
        # Initialize grid layout
        layout = QGridLayout()

        # Create horizontal spacers
        h_spacer1 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)  # Left spacer
        h_spacer2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)  # Right spacer

        # Create vertical spacer
        v_spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        # Add horizontal spacers and summary label to the top row
        layout.addItem(h_spacer1, 0, 0)  # Left spacer
        layout.addWidget(summary_label, 0, 1)  # Summary label
        layout.addItem(h_spacer2, 0, 2)  # Right spacer

        spending_labels_layout = QHBoxLayout()

        current_month_widget = SpendingLabel("Current Month", f"${mtd_spending}")
        ytd_spending_widget = SpendingLabel("Year-to-Date", f"${ytd_spending}")

        # Add the SpendingLabel widgets to the QHBoxLayout
        spending_labels_layout.addWidget(current_month_widget)
        spending_labels_layout.addWidget(ytd_spending_widget)

        # Add the QHBoxLayout to the main QGridLayout
        layout.addLayout(spending_labels_layout, 1, 0, 1, 3)

        # Position mtd_web_view on the left (column 0) and web_view on the right (column 2)
        layout.addWidget(self.mtd_web_view, 3, 0, 1, 1)  # Spanning 1 row and 1 column
        layout.addWidget(self.web_view, 3, 2, 1, 1)      # Spanning 1 row and 1 column

        # Add a vertical spacer at the bottom
        layout.addItem(v_spacer, 4, 0, 1, 3)

        # Set the layout for the widget
        self.setLayout(layout)

    @pyqtSlot()
    def update_theme(self):
        ThemeManager.update_widget_theme(self)

    def update_content(self):
        self.update_graph()
        self.update_mtd_graph()

    def update_graph(self):
        fig = self.graph_logic.ytd_cat_subcat_sunburst()
        bg_color = ThemeManager.get_plotly_bgcolor()
        fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
        )
        config = {'displayModeBar': False}
        raw_html = pio.to_html(fig, include_plotlyjs=False, full_html=False, config=config)
        theme_css = ThemeManager.get_webview_css()
        raw_html_with_cdn = f'<html style="background: transparent;"><head>{theme_css}</head><body><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>{raw_html}</body></html>'
        self.web_view.setHtml(raw_html_with_cdn)
        self.web_view.page().runJavaScript('location.reload();')
        
    def update_mtd_graph(self):
        mtd_fig = self.graph_logic.mtd_cat_subcat_sunburst()
        bg_color = ThemeManager.get_plotly_bgcolor()
        mtd_fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
        )
        config = {'displayModeBar': False}
        mtd_raw_html = pio.to_html(mtd_fig, include_plotlyjs=False, full_html=False, config=config)
        theme_css = ThemeManager.get_webview_css()
        mtd_raw_html_with_cdn = f'<html style="background: transparent;"><head>{theme_css}</head><body><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>{mtd_raw_html}</body></html>'
        self.mtd_web_view.setHtml(mtd_raw_html_with_cdn)
        self.mtd_web_view.page().runJavaScript('location.reload();')



    def set_up_graph(self, web_view, fig):
        config = {'displayModeBar': False}
        bg_color = ThemeManager.get_plotly_bgcolor()
        fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
        )
        raw_html = pio.to_html(fig, include_plotlyjs=False, full_html=False, config=config)
        theme_css = ThemeManager.get_webview_css()
        raw_html_with_cdn = f'<html style="background: transparent;"><head>{theme_css}</head><body><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>{raw_html}</body></html>'
        web_view.setHtml(raw_html_with_cdn)
