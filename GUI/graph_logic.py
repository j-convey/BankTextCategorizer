import pandas as pd
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from decimal import Decimal


categories = {
            'Auto': ['Gas','Maintenance', 'Upgrades', 'Other_Auto'],
            'Baby': ['Diapers', 'Formula', 'B_Clothes', 'Toys', 'Other_Baby'],
            'Clothes': ['Clothes', 'Shoes', 'Jewelry', 'Bags_Accessories'],
            'Entertainment': ['Sports_Outdoors', 'Movies_TV', 'DateNights', 'Arts_Crafts', 'Books', 'Games', 'Guns', 'E_Other'],
            'Electronics': ['Accessories', 'Computer', 'TV', 'Camera', 'Phone','Tablet_Watch', 'Gaming', 'Electronics_misc'],
            'Food': ['Groceries', 'FastFood_Restaurants'],
            'Home': ['Rent', 'Maintenance', 'Furniture_Appliances', 'Hygiene', 'Gym',
                'Home_Essentials', 'Kitchen', 'Decor', 'Security', 'Yard_Garden', 'Tools'],
            'Medical': ['Health_Wellness'],
            'Kids': ['K_Toys'],
            'Personal_Care': ['Hair', 'Makeup_Nails', 'Beauty', 'Massage','Vitamins_Supplements', 'PC_Other'],
            'Pets': ['Pet_Food', 'Pet_Toys', 'Pet_Med', 'Pet_Grooming', 'Pet_Other'],
            'Subscriptions_Memberships': ['Entertainment', 'Gym', 'Sub_Other'],
            'Travel': ['Hotels', 'Flights', 'Car_Rental', 'Activities']}

class GraphLogic():
    def __init__(self):
        self.df = pd.read_csv('GUI/GraphTest.csv')
        self.df['Date'] = pd.to_datetime(self.df['Date'])  # Convert the 'Date' column to datetime format
    def return_df(self):
        return self.df
    def save_df(self):
        self.df.to_csv('GUI/GraphTest.csv', index=False)

    def return_monthyear(self):
        current_month = datetime.datetime.now().month
        current_year = datetime.datetime.now().year
        return current_month, current_year

    def mtd_spending(self):
        current_month, current_year = self.return_monthyear()
        # Filter the DataFrame to only include rows where the month and year match
        current_month_df = self.df[
            (self.df['Date'].dt.month == current_month) & 
            (self.df['Date'].dt.year == current_year)]
        # Use Decimal for more precise arithmetic
        return Decimal(current_month_df['Price'].sum()).quantize(Decimal("0.00"))
    
    def ytd_spending(self):
        current_month, current_year = self.return_monthyear()
        current_year_df = self.df[(self.df['Date'].dt.year == current_year)]
        return current_year_df['Price'].sum()
    
    
    def ytd_cat_subcat_sunburst(self):
        current_year = datetime.datetime.now().year
        current_year_df = self.df[self.df['Date'].dt.year == current_year]

        labels = []
        parents = []
        values = []

        for category, sub_categories in categories.items():
            category_data = current_year_df[current_year_df['Category'] == category]
            category_value = category_data['Price'].sum()
            #print(f"Category: {category}, Value: {category_value}")
            if category_value == 0:
                continue
            labels.append(str(category))
            parents.append('')
            values.append(float(category_value))
            for sub_category in sub_categories:
                sub_category_data = category_data[category_data['Sub_category'] == sub_category]
                sub_category_value = sub_category_data['Price'].sum()
                #print(f"  Sub-category: {sub_category}, Value: {sub_category_value}")
                if sub_category_value == 0:
                    continue
                unique_label = f"{sub_category} (sub)" if sub_category == category else sub_category
                labels.append(str(unique_label))
                parents.append(str(category))
                values.append(float(sub_category_value))
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values))
        fig.update_layout(
            title="YTD", 
            margin=dict(l=0, r=0, b=0, t=0),
            width=480,  # set the width
            height=280)  # set the height
        #fig.show()
        return fig
    
    def mtd_cat_subcat_sunburst(self):
        current_date = datetime.datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        current_month_df = self.df[
            (self.df['Date'].dt.year == current_year) & 
            (self.df['Date'].dt.month == current_month)
        ]

        labels = []
        parents = []
        values = []

        for category, sub_categories in categories.items():
            category_data = current_month_df[current_month_df['Category'] == category]
            category_value = category_data['Price'].sum()
            if category_value == 0:
                continue
            labels.append(str(category))
            parents.append('')
            values.append(float(category_value))
            for sub_category in sub_categories:
                sub_category_data = category_data[category_data['Sub_category'] == sub_category]
                sub_category_value = sub_category_data['Price'].sum()
                if sub_category_value == 0:
                    continue
                unique_label = f"{sub_category} (sub)" if sub_category == category else sub_category
                labels.append(str(unique_label))
                parents.append(str(category))
                values.append(float(sub_category_value))

        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values
        ))

        fig.update_layout(
            title=f"MTD for {datetime.date(current_year, current_month, 1).strftime('%B')}", 
            margin=dict(l=0, r=0, b=0, t=0),
            width=480,  # set the width
            height=280  # set the height
        )
        return fig



'''obj = GraphLogic()
print(obj.return_monthyear())
print(obj.mtd_spending())
print(obj.ytd_spending())
obj.ytd_cat_subcat_sunburst()'''
