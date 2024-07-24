import os
import joblib # for saving the `queue` encoder
from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch

df = pd.read_csv(os.getcwd() + '/ticket-helpdesk-multi-lang.csv')

# Preprocessing
df['text'] = df['text'].str.lower()

# Encode the `queue` column
label_encoder = LabelEncoder()
df['queue_encoded'] = label_encoder.fit_transform(df['queue'])

# Save the label encoder
joblib.dump(label_encoder, 'queue_encoder.joblib')

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['queue_encoded'].tolist(),
    test_size=0.2,
    random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("cwchang/text-classification-model-multilingual")
model = AutoModelForSequenceClassification.from_pretrained("cwchang/text-classification-model-multilingual",
                                                           num_labels = df['queue'].nunique(),
                                                           ignore_mismatched_sizes = True)

# Tokenize the text
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create a dataset object
class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = TicketDataset(train_encodings, train_labels)
val_dataset = TicketDataset(val_encodings, val_labels)

optimizer = AdamW(model.parameters(), lr=1e-5)

# Set up the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
model.to(device)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'Train loss: {train_loss / len(train_loader)}')

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    print(f'Validation loss: {val_loss / len(val_loader)}')

model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
