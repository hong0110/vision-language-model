import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import editdistance

# ===== CONFIG =====
TRAIN_PATH = "/content/data/train"
VAL_PATH = "/content/data/val"
OUTPUT_DIR = "/content/output/model"

BATCH_SIZE = 1   # giảm để tránh OOM
EPOCHS = 5
LR = 3e-5
MAX_LENGTH = 512

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===== LOAD MODEL =====
print("Loading model...")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# FIX lỗi pad_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id

# giảm RAM
model.config.use_cache = False
model.gradient_checkpointing_enable()

model.to(device)


# ===== DATASET =====
class InvoiceDataset(Dataset):
    def __init__(self, path):
        with open(os.path.join(path, "annotations.json"), "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.path = path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = os.path.join(self.path, "images", item["image"])
        image = Image.open(image_path).convert("RGB")

        pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()

        text = item["ground_truth"]
        labels = processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).input_ids.squeeze()

        # ignore pad token
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return pixel_values, labels, text


# ===== LOAD DATA =====
train_dataset = InvoiceDataset(TRAIN_PATH)
val_dataset = InvoiceDataset(VAL_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")


# ===== OPTIMIZER =====
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# ===== METRIC =====
def extract_tag(text, tag):
    import re
    match = re.search(f"<{tag}>(.*?)</{tag}>", text)
    return match.group(1) if match else ""


# ===== TRAIN LOOP =====
train_losses = []
val_losses = []
val_edit_distances = []

for epoch in range(EPOCHS):
    print(f"\n===== EPOCH {epoch+1} =====")

    # ===== TRAIN =====
    model.train()
    total_loss = 0

    for pixel_values, labels, _ in tqdm(train_loader):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Train Loss: {avg_train_loss:.4f}")


    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    total_edit = 0

    with torch.no_grad():
        for pixel_values, labels, gt_text in val_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            # generate
            generated_ids = model.generate(pixel_values, max_length=MAX_LENGTH)
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # edit distance
            ed = editdistance.eval(pred, gt_text[0])
            total_edit += ed

    avg_val_loss = val_loss / len(val_loader)
    avg_edit = total_edit / len(val_loader)

    val_losses.append(avg_val_loss)
    val_edit_distances.append(avg_edit)

    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Edit Distance: {avg_edit:.4f}")


# ===== SAVE MODEL =====
print("Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)


# ===== PLOT LOSS =====
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# ===== PLOT EDIT DISTANCE =====
plt.figure()
plt.plot(val_edit_distances, label="Edit Distance")
plt.legend()
plt.title("Edit Distance per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Distance")
plt.show()


print("DONE TRAINING")