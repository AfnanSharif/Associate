
import pandas as pd
import numpy as np
import yaml
import os

def generate_data(num_samples: int, output_path: str):
    import random
    texts = ["Great product!", "Horrible experience.", "Average quality, okay.", "I loved this!", "Waste of money."]
    labels = ["Positive", "Negative", "Neutral", "Positive", "Negative"]
    data = [(random.choice(texts), random.choice(labels)) for _ in range(num_samples)]
    df = pd.DataFrame(data, columns=['text', 'label'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    generate_data(config['data']['num_samples'], config['data']['raw_path'])
