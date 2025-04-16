import torch
from tqdm import tqdm

from data import Data, data_collator
from preprocess import (load_poi_categories, preprocess_stays,
                        read_parquet_recursive)
from vambc import Vambc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def cluster(model, dataloader, device):
    model.eval()
    pbar = tqdm(dataloader, desc='Testing')
    reprs, assignments = [], []
    for x in pbar:
        x = x.to(device)
        with torch.no_grad():
            y, zb, xp, xc, z = model(x)
        assignment = y.argmax(dim=-1)
        assignments.append(assignment)
        reprs.append(z)
    pbar.close()
    assignments = torch.cat(assignments, dim=0)
    reprs = torch.cat(reprs, dim=0)
    return reprs, assignments

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load stay points
    data_path = '/home/Shared/novateur.phase2.trial4/stay_poi_dfs'
    stay_df = read_parquet_recursive(data_path)
    stay_df = preprocess_stays(stay_df)

    # Load POI categories
    poi2vec = load_poi_categories()
    poi2vec = torch.from_numpy(poi2vec).float()
    dim_x = poi2vec.shape[1]
    print(f'POI vector size: {dim_x}')

    # Get the index of each agent's first and last row
    stay_df['index'] = stay_df.index
    agents = stay_df.groupby('agent').agg(first=('index', 'first'), last=('index', 'last'))

    # Split agents
    test_agents = agents.sample(frac=1/100, random_state=42)
    # test_agents = agents

    # Prepare data
    test_data = Data(stay_df, poi2vec, test_agents['first'], test_agents['last'])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=data_collator)

    model = Vambc(dim_x, dim_lstm=128, dim_z=64, zb_std=1, num_clusters=10).to(device)
    model.load_state_dict(torch.load('output/model.pth'))
    
    reprs, assignments = cluster(model, test_loader, device)

    # Visualize clusters
    # Use t-sne for xc
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    repr_tsne = tsne.fit_transform(reprs.cpu().numpy())
    # Plot
    plt.scatter(repr_tsne[:, 0], repr_tsne[:, 1], c=assignments.cpu().numpy())
    plt.savefig('output/cluster.png')
