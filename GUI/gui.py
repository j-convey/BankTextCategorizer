import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd

class Application(tk.Tk):
    
    def __init__(self):
        super().__init__()
        
        self.title("Bank Description Categorizer")
        self.geometry("600x400")
        
        # Variables
        self.csv_files = []

        # Model Path
        self.model_label = tk.Label(self, text="Model Path:")
        self.model_label.grid(row=0, column=0, padx=10, pady=10)

        self.model_entry = tk.Entry(self, width=50)
        self.model_entry.grid(row=0, column=1, padx=10, pady=10)

        self.model_button = tk.Button(self, text="Browse", command=self.load_model_path)
        self.model_button.grid(row=0, column=2, padx=10, pady=10)

        # CSV Files
        self.csv_label = tk.Label(self, text="CSV Files:")
        self.csv_label.grid(row=1, column=0, padx=10, pady=10)

        self.csv_listbox = tk.Listbox(self, width=50)
        self.csv_listbox.grid(row=1, column=1, padx=10, pady=10)

        self.csv_button = tk.Button(self, text="Add CSV", command=self.add_csv_path)
        self.csv_button.grid(row=1, column=2, padx=10, pady=10)

        # Merge CSVs
        self.merge_button = tk.Button(self, text="Merge CSVs", command=self.merge_csvs)
        self.merge_button.grid(row=2, column=1, padx=10, pady=20)

        # Process
        self.process_button = tk.Button(self, text="Process", command=self.process_data)
        self.process_button.grid(row=3, column=1, padx=10, pady=20)

    def load_model_path(self):
        file_path = filedialog.askopenfilename()
        self.model_entry.delete(0, tk.END)
        self.model_entry.insert(0, file_path)

    def add_csv_path(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.csv_files.append(file_path)
            self.csv_listbox.insert(tk.END, file_path)

    def merge_csvs(self):
        if len(self.csv_files) < 2 or len(self.csv_files) > 8:
            messagebox.showerror("Error", "Select between 2 to 8 CSV files for merging.")
            return

        try:
            merged_df = self.merge_csv_files(*self.csv_files)
            # Here you can choose to save the merged_df to a file or use it elsewhere
            messagebox.showinfo("Success", "CSV files merged successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

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
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
