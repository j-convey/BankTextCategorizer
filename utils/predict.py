import csv
import torch
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

class Predict:
    def __init__(self, loaded_category_model, predict_dataloader, csv_output_name):
        """Initialize the Predict class with model, dataloader, device, and output file name."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = loaded_category_model.to(self.device)
        self.dataloader = predict_dataloader
        self.csv_output_name = csv_output_name
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_encoder_cat = LabelEncoder()


    def run_prediction(self):
        """Run predictions on the data and save results to a CSV file."""
        with open(self.csv_output_name, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Description', 'Category', 'Subcategory']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()

            for batch in self.dataloader:
                input_ids = batch[0].to(self.device)
                with torch.no_grad():
                    category_probs, _ = self.model(input_ids)
                    category_predictions = category_probs.argmax(dim=-1)

                for i in range(input_ids.size(0)):
                    category_name = self.label_encoder_cat.inverse_transform([category_predictions[i].item()])[0]
                    single_input_ids = input_ids[i].to('cpu')
                    tokens = self.tokenizer.convert_ids_to_tokens(single_input_ids)
                    description = self.tokenizer.convert_tokens_to_string(tokens).strip()
                    writer.writerow({'Description': description, 'Category': category_name})
        
        print(f"Predictions saved to {self.csv_output_name}")