import torch
from tqdm import tqdm

from data import Data, data_collator
from preprocess import (load_poi_categories, preprocess_stays,
                        read_parquet_recursive)
from vambc import Vambc, vambc_loss


def train(model, optimizer, dataloader, device):
    model.train()
    train_loss = []
    pbar = tqdm(dataloader, desc='Training')
    for x in pbar:
        x = x.to(device)

        optimizer.zero_grad()
        y, zb, xc, xp, z = model(x)
        loss = vambc_loss(x, y, zb, xc, xp)
        loss.backward()
        optimizer.step()

        train_loss += [loss.item()]
        pbar.set_postfix(loss=loss.item())
    pbar.close()
    return train_loss


def evaluate(model, dataloader, device):
    model.eval()
    val_loss = []
    pbar = tqdm(dataloader, desc='Evaluating')
    for x in pbar:
        x = x.to(device)
        with torch.no_grad():
            y, zb, xp, xc, z = model(x)

        loss = vambc_loss(x, y, zb, xc, xp)
        val_loss += [loss.item()]
        pbar.set_postfix(loss=loss.item())
    pbar.close()

    val_loss = sum(val_loss) / len(val_loss)
    return val_loss


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
    test_agents = agents.sample(frac=1/10, random_state=42)
    train_agents = agents.drop(test_agents.index).sample(frac=8/9, random_state=42)
    val_agents = agents.drop(test_agents.index).drop(train_agents.index)

    # Prepare data
    train_data = Data(stay_df, poi2vec, train_agents['first'], train_agents['last'])
    val_data = Data(stay_df, poi2vec, val_agents['first'], val_agents['last'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=data_collator)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=data_collator)

    model = Vambc(dim_x, dim_lstm=128, dim_z=64, zb_std=1, num_clusters=10).to(device)
    optimizer = torch.optim.Adafactor(model.parameters())

    best_val_loss = float('inf')
    for epoch in range(10):
        train_loss = train(model, optimizer, train_loader, device)
        val_loss = evaluate(model, val_loader, device)
        print(f'Epoch {epoch}, Train Loss: {sum(train_loss) / len(train_loss)}, Val Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'output/model.pth')
            