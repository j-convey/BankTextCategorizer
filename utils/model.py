import torch.nn as nn
from transformers import BertForSequenceClassification

class BertModel(nn.Module):
    def __init__(self, num_categories, num_subcategories):
        super().__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels= num_categories + num_subcategories)      
        self.num_categories = num_categories
        self.num_subcategories = num_subcategories
        
    def forward(self, input_ids):
        outputs = self.bert_model(input_ids)
        logits = outputs.logits
        category_logits, subcategory_logits = logits.split([self.num_categories, self.num_subcategories], dim=-1)
        return category_logits, subcategory_logits
    
# def execute_cat_model(cat_model, cat_train_dataloader, cat_val_dataloader, device, num_categories, learning_rate, epochs):
#     '''Category Training & Saving'''    
#     cat_model.to(device)
#     category_history = train_model(cat_model, cat_train_dataloader, cat_val_dataloader, epochs, learning_rate, device, num_categories)
#     # Move the model back to CPU before saving
#     cat_model.to('cpu')
#     cat_model_save_path = 'models/pt_cat_modelV1'
#     torch.save(cat_model.state_dict(), cat_model_save_path)
#     plot_training_history(category_history)

# def execute_sub_model(sub_model, sub_train_dataloader, sub_val_dataloader, device, num_subcategories, learning_rate, epochs):
#     '''Subcategory Training & Saving'''
#     sub_model.to(device)
#     subcategory_history = train_model(sub_model, 'subcategory', sub_train_dataloader, sub_val_dataloader, epochs, learning_rate, device)
#     sub_model.to('cpu')
#     sub_model_save_path = 'models/pt_sub_modelV1'
#     torch.save(sub_model.state_dict(), sub_model_save_path)
#     plot_training_history(subcategory_history)

