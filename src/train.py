
import pandas as pd
import yaml
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline

def train_model():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    df = pd.read_csv(config['data']['raw_path'])
    pipeline = Pipeline([('tfidf', TfidfVectorizer(max_df=0.9, min_df=1)), ('nmf', NMF(n_components=2, random_state=42))])
    pipeline.fit(df['text'])
    os.makedirs(os.path.dirname(config['data']['model_path']), exist_ok=True)
    joblib.dump(pipeline, config['data']['model_path'])

if __name__ == "__main__":
    train_model()
