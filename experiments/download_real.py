"""Download real images using urllib (no HF datasets dependency)."""
import os
import urllib.request
import json

real_dir = "data/aigenbench/real"
os.makedirs(real_dir, exist_ok=True)

# Use picsum.photos for diverse real photographs
print("Downloading 30 real photographs...")
for i in range(30):
    url = f"https://picsum.photos/seed/{i+100}/512/512"
    path = os.path.join(real_dir, f"real_{i:04d}.jpg")
    if os.path.exists(path):
        continue
    try:
        urllib.request.urlretrieve(url, path)
        if (i + 1) % 10 == 0:
            print(f"  Downloaded {i+1}/30")
    except Exception as e:
        print(f"  Failed {i}: {e}")

n = len([f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))])
print(f"Done: {n} real images in {real_dir}")
