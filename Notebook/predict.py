!pip install PyDrive
!pip install transformers


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

def predict_scy(text, model, tokenizer, max_length=512, device='cpu'):
    # Prepare input tokens
    input_tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move tensors to the specified device
    input_ids = input_tokens['input_ids'].to(device)
    attention_mask = input_tokens['attention_mask'].to(device)

    # Get model output
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get the predicted label index
    logits = outputs.logits
    predicted_index = torch.argmax(logits, dim=1).item()

    return predicted_index




# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

folder_id = '12oNKgmSCNRdGfUS1oxhEhsXn-1V_yC0v'  # ID of the folder
file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

# Find and download the model file
model_file_name = 'my_model_resampled_data.pt'
model_file_id = None
for file in file_list:
    if file['title'] == model_file_name:
        model_file_id = file['id']
        break
download_file = drive.CreateFile({'id': model_file_id})
download_file.GetContentFile(model_file_name)

# Load the saved model's state dictionary
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load(model_file_name))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
