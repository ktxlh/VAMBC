import os

import pandas as pd


def convert_pois_to_catetories():
    # Load data and drop unnecessary columns
    poi_df = pd.read_csv(
        '/home/Shared/novateur.phase2.trial4/poi.csv', 
        usecols=['poi_id', 'category']
    )

    # One-hot encode categories directly
    poi_cat_vecs = pd.get_dummies(poi_df.set_index('poi_id')['category']).groupby('poi_id').max()

    # Save to parquet
    poi_cat_vecs.to_parquet('/home/Shared/novateur.phase2.trial4/poi_cat_vecs.parquet')


def load_poi_categories():
    poi2vec = pd.read_parquet('/home/Shared/novateur.phase2.trial4/poi_cat_vecs.parquet').values
    print(f'Number of POIs: {len(poi2vec):,}')
    return poi2vec

def read_parquet_recursive(dir):
    data = []
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isdir(path):
            data.append(read_parquet_recursive(path))
        elif path.endswith('.parquet'):
            data.append(pd.read_parquet(path))
    return pd.concat(data)


def preprocess_stays(stay_df):
    # Set poi_id to int
    stay_df['poi_id'] = stay_df['poi_id'].astype(int)
    stay_df.reset_index(drop=True, inplace=True)
    
    print(f'Number of stay points: {len(stay_df):,}')
    print(f'Number of agents: {stay_df.agent.nunique():,}')
    return stay_df