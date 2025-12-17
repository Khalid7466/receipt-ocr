from datasets import load_dataset
import os

# 1. Download Official Data
print("⏳ Downloading CORD v2 from Hugging Face...")
dataset = load_dataset("naver-clova-ix/cord-v2")

# 2. Save paths
base_path = "data/cord/raw"

def save_split(split_name):
    print(f"📂 Saving {split_name} data...")
    split_dir = os.path.join(base_path, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for i, item in enumerate(dataset[split_name]):
        # Save Image
        image_path = os.path.join(split_dir, f"image_{i}.png")
        item['image'].save(image_path)

        # Save Label (JSON)
        import json
        json_path = os.path.join(split_dir, f"image_{i}.json")
        with open(json_path, 'w') as f:
            json.dump(item['ground_truth'], f)

# Run for all splits
save_split("train")
save_split("validation") # CORD uses 'validation' instead of 'dev' sometimes
save_split("test")
print("✅ Done! CORD dataset saved to data/cord/raw")