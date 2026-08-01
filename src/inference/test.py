import json
with open("fruit-image-classification/artifacts/class_to_idx.json", "r", encoding="utf-8") as f:
       class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
print(idx_to_class)