import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import CLIPProcessor
from src.model import CLIPClassifier
from src.data_prep import load_and_preprocess

class FakeNewsDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        image = item['image']
        label = item['label']

        inputs = self.processor(
            text=text,
            images=image if image is not None else Image.new("RGB", (224, 224), color='white'),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label)
        return inputs

# Load data
csv_path = "data/raw/Cleaned_news_no_Unknown.csv"
image_dir = "data/raw/images"
data = load_and_preprocess(csv_path, image_dir)

# Prepare data loader
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
dataset = FakeNewsDataset(data, processor)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model and train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        logits, _ = model(**batch)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(data_loader):.4f}")

# Save model
torch.save(model.state_dict(), "models/clip_fake_news_classifier1.pth")
print(" Model training complete and saved.")
