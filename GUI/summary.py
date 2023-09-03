from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QSizePolicy, QSpacerItem, QSizePolicy
from PyQt6.QtCore import Qt, QSize, pyqtSlot
from PyQt6.QtWebEngineWidgets import QWebEngineView
import datetime
import plotly.graph_objects as go
import plotly.io as pio
from graph_logic import GraphLogic
from message_box import ThemeManager

class SummaryTab(QWidget):
    def __init__(self):
        super(SummaryTab, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.graph_logic = GraphLogic()
        self.web_view = QWebEngineView()
        self.web_view.setMaximumSize(QSize(600, 400))

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
        fig = self.graph_logic.ytd_cat_subcat_sunburst()
        # Configuration to remove mode bar (toolbar)
        fig.update_layout(
            paper_bgcolor=ThemeManager.get_plotly_bgcolor(),  # background from theme
            plot_bgcolor=ThemeManager.get_plotly_bgcolor(),  # background from theme
        )

        config = {'displayModeBar': False}
        # Convert the Plotly figure to HTML
        raw_html = pio.to_html(fig, include_plotlyjs=False, full_html=False, config=config)
        # Create a QWebEngineView object
        self.web_view = QWebEngineView()

        # Set a fixed size or maximum size if you want to control the dimensions
        self.web_view.setMaximumSize(QSize(600, 400))

        # Include your CSS for the current theme in the HTML
        theme_css = ThemeManager.get_webview_css()


        # Add the CSS to your existing HTML
        raw_html_with_cdn = f'<html style="background: transparent;"><head>{theme_css}</head><body><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>{raw_html}</body></html>'
        self.web_view.setHtml(raw_html_with_cdn)
        self.web_view.setMaximumSize(QSize(600, 400))

        summary_label.setMaximumHeight(40)
        current_month_label.setMaximumHeight(40)
        ytd_spending_label.setMaximumHeight(40)
        #ytd_spending label to not expand
        ytd_spending_label.setProperty("class", "no_expand")
        # Create a QGridLayout
        layout = QGridLayout()
        # Create spacers
        v_spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        h_spacer1 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)  # Left spacer
        h_spacer2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)  # Right spacer

        # Add widgets and spacers to layout
        layout.addItem(h_spacer1, 0, 0)  # Left spacer
        layout.addWidget(summary_label, 0, 1)  # Summary label
        layout.addItem(h_spacer2, 0, 2)  # Right spacer

        layout.addWidget(current_month_label, 1, 0, 1, 3)  # Spanning 1 row and 3 columns
        layout.addWidget(ytd_spending_label, 2, 0, 1, 3)  # Spanning 1 row and 3 columns
        layout.addWidget(self.web_view, 3, 0, 1, 3)  # Spanning 1 row and 3 columns

        # Add vertical spacer
        layout.addItem(v_spacer, 4, 0, 1, 3)

        self.setLayout(layout)
    @pyqtSlot()
    def update_theme(self):
        ThemeManager.update_widget_theme(self)
    def update_content(self):
        self.update_graph()

    def update_graph(self):
        fig = self.graph_logic.ytd_cat_subcat_sunburst()
        bg_color = ThemeManager.get_plotly_bgcolor()
        print(f"Theme background color: {bg_color}")  # Debug statement 2
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
