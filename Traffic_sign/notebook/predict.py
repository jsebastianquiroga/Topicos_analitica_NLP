{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPNQ/6ppAys1jUckBzSK+9R"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZlcAtywhAYgq"
      },
      "outputs": [],
      "source": [
        "!pip install PyDrive\n",
        "!pip install transformers\n",
        "\n",
        "\n",
        "from pydrive.auth import GoogleAuth\n",
        "from pydrive.drive import GoogleDrive\n",
        "from google.colab import auth\n",
        "from oauth2client.client import GoogleCredentials\n",
        "import os\n",
        "from transformers import BertTokenizer, BertForSequenceClassification\n",
        "import torch\n",
        "\n",
        "import torch\n",
        "from torch.utils.data import DataLoader\n",
        "from transformers import BertTokenizer, BertForSequenceClassification\n",
        "\n",
        "def predict_scy(text, model, tokenizer, max_length=512, device='cpu'):\n",
        "    # Prepare input tokens\n",
        "    input_tokens = tokenizer.encode_plus(\n",
        "        text,\n",
        "        add_special_tokens=True,\n",
        "        max_length=max_length,\n",
        "        padding='max_length',\n",
        "        truncation=True,\n",
        "        return_attention_mask=True,\n",
        "        return_tensors='pt'\n",
        "    )\n",
        "\n",
        "    # Move tensors to the specified device\n",
        "    input_ids = input_tokens['input_ids'].to(device)\n",
        "    attention_mask = input_tokens['attention_mask'].to(device)\n",
        "\n",
        "    # Get model output\n",
        "    with torch.no_grad():\n",
        "        outputs = model(input_ids, attention_mask=attention_mask)\n",
        "    \n",
        "    # Get the predicted label index\n",
        "    logits = outputs.logits\n",
        "    predicted_index = torch.argmax(logits, dim=1).item()\n",
        "\n",
        "    return predicted_index\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "# Authenticate and create the PyDrive client\n",
        "auth.authenticate_user()\n",
        "gauth = GoogleAuth()\n",
        "gauth.credentials = GoogleCredentials.get_application_default()\n",
        "drive = GoogleDrive(gauth)\n",
        "\n",
        "folder_id = '12oNKgmSCNRdGfUS1oxhEhsXn-1V_yC0v'  # ID of the folder\n",
        "file_list = drive.ListFile({'q': f\"'{folder_id}' in parents and trashed=false\"}).GetList()\n",
        "\n",
        "# Find and download the model file\n",
        "model_file_name = 'my_model_resampled_data.pt'\n",
        "model_file_id = None\n",
        "for file in file_list:\n",
        "    if file['title'] == model_file_name:\n",
        "        model_file_id = file['id']\n",
        "        break\n",
        "download_file = drive.CreateFile({'id': model_file_id})\n",
        "download_file.GetContentFile(model_file_name)\n",
        "\n",
        "# Load the saved model's state dictionary\n",
        "model = BertForSequenceClassification.from_pretrained('bert-base-uncased')\n",
        "model.load_state_dict(torch.load(model_file_name))\n",
        "\n",
        "tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')"
      ]
    }
  ]
}