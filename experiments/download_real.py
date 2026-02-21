"""Download real images from tiny-imagenet for light estimation."""
from datasets import load_dataset
import os

os.makedirs("data/aigenbench/real", exist_ok=True)
ds = load_dataset("zh-plus/tiny-imagenet", split="valid", streaming=True)
count = 0
for sample in ds:
    if count >= 30:
        break
    img = sample["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((512, 512))
    img.save(f"data/aigenbench/real/real_{count:04d}.png")
    count += 1
    if count % 10 == 0:
        print(f"Downloaded {count}/30")
print(f"Done: {count} real images")
