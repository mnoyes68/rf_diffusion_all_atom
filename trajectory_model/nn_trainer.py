import argparse
import json
import os
import sys
import numpy as np
import torch
import json

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from torch import Tensor
from torch.nn import Dropout, Linear, ReLU, Sequential
from torch.nn.init import kaiming_uniform_, xavier_uniform_, zeros_
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import GeometricShapes
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, MessagePassing
from torch_geometric.transforms import Compose, KNNGraph, RadiusGraph, SamplePoints


class BackboneDataset(InMemoryDataset):
    def __init__(self, input_data: list[dict], transform=RadiusGraph(r=4)):
        super().__init__(transform=transform)

        data_list = []
        for d in input_data:
            data_list.append(
                Data(
                    pos=torch.tensor(d['coords'], dtype=torch.float),
                    y=torch.tensor(d['ptm'], dtype=torch.float),
                    meta=torch.tensor(d['model_id']/200, dtype=torch.float)
                )
            )

        self.data, self.slices = self.collate(data_list)


class BackbonePointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PointNetLayer(3, 64)
        self.conv2 = PointNetLayer(64, 64)
        self.conv3 = PointNetLayer(64, 128)
        

        l1 = Linear(128, 64)
        l2 = Linear(64, 1)
        xavier_uniform_(l1.weight)
        xavier_uniform_(l2.weight)
        if l1.bias is not None:
            zeros_(l1.bias)
        if l2.bias is not None:
            zeros_(l2.bias)
        self.regressor = Sequential(
            l1,
            Dropout(0.5),
            l2
        )

    def forward(self,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        timestep: Tensor,
    ) -> Tensor:        
        # Perform three layers of message passing:
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # Global Pooling:
        h = global_mean_pool(h, batch)  # [num_examples, hidden_channels]

        # Classifier:
        return torch.sigmoid(self.regressor(h)).squeeze(-1)


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        l1 = Linear(in_channels + 3, out_channels)
        l2 = Linear(out_channels, out_channels)

        kaiming_uniform_(l1.weight, nonlinearity='relu')
        kaiming_uniform_(l2.weight, nonlinearity='relu')
        if l1.bias is not None:
            zeros_(l1.bias)
        if l2.bias is not None:
            zeros_(l2.bias)

        self.mlp = Sequential(l1, ReLU(), l2)

    def forward(self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self,
        h_j: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
    ) -> Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


def train(
    model: BackbonePointNet,
    data_loader: DataLoader,
    device: str,
    optimizer,
    criterion
):
    model.train()
    total_loss = 0.0

    for data in tqdm(data_loader, desc="Training"):
        # Setup
        optimizer.zero_grad()
        data.to(device)

        # Train
        pred = model(data.pos, data.edge_index, data.batch, data.meta)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()

        # Track stats
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(data_loader.dataset)


@torch.no_grad()
def test(model: BackbonePointNet, data_loader: DataLoader, device: str):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0

    for data in tqdm(data_loader, desc="Testing"):
        data.to(device)
        pred = model(data.pos, data.edge_index, data.batch, data.meta)
        loss = criterion(pred, data.y)

        # Track stats
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(data_loader.dataset)


def main():
    # Load args
    parser = argparse.ArgumentParser(description="A simple command-line tool.")
    parser.add_argument("input_data", type=str)
    parser.add_argument("output_directory", type=str)
    parser.add_argument("-ep", "--epochs", type=int, default=50)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.01)
    parser.add_argument("-bs", "--batch-size", type=int, default=128)
    parser.add_argument("-r", "--radius", type=float, default=4)
    parser.add_argument("-fp", "--from-pretrained", type=str, default=None)
    args = parser.parse_args()

    # Setup and load data
    os.makedirs(args.output_directory, exist_ok=True)
    data = np.load(args.input_data, allow_pickle=True)

    # Train Test Split, grouped by backbone
    backbones = list(set([item['backbone'] for item in data]))
    if args.from_pretrained:
        # Load from the original values in from_pretrained
        train_backbones = np.load(os.path.join(args.from_pretrained, "train_backbones.npy"))
        test_backbones = np.load(os.path.join(args.from_pretrained, "test_backbones.npy"))
        val_backbones = np.load(os.path.join(args.from_pretrained, "val_backbones.npy"))
    else:
        # Run a train test validation split to get the data
        train_backbones, test_val_backbones = train_test_split(backbones, train_size=0.8)
        test_backbones, val_backbones = train_test_split(test_val_backbones, train_size=0.5)
        np.save(os.path.join(args.output_directory, "train_backbones.npy"), train_backbones)
        np.save(os.path.join(args.output_directory, "test_backbones.npy"), test_backbones)
        np.save(os.path.join(args.output_directory, "val_backbones.npy"), val_backbones)

    train_backbone_set = set(train_backbones)
    test_backbone_set = set(test_backbones)
    val_backbones_set = set(val_backbones)
    train_data = []
    test_data = []
    val_data = []
    for d in data:
        if d['backbone'] in train_backbone_set:
            train_data.append(d)
        elif d['backbone'] in test_backbone_set:
            test_data.append(d)
        elif d['backbone'] in val_backbones_set:
            val_data.append(d)
        else:
            raise ValueError(f"Cannot find backbone {d['backbone']} in either dataset.")

    train_dataset = BackboneDataset(train_data, transform=RadiusGraph(r=args.radius))
    test_dataset = BackboneDataset(test_data, transform=RadiusGraph(r=args.radius))
    val_dataset = BackboneDataset(val_data, transform=RadiusGraph(r=args.radius))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BackbonePointNet().to(device)
    if args.from_pretrained:
        print("Loading model from weights")
        state_dict = torch.load(os.path.join(args.from_pretrained, 'model_weights.pth'))
        model.load_state_dict(state_dict)

    # Training
    train_loss_arr = []
    test_loss_arr = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    min_loss = 1.0
    epoch_without_improvement = 0
    patience = 5
    for epoch in range(args.epochs):
        train_loss = train(model, train_data_loader, device, optimizer, criterion)
        train_loss_arr.append(train_loss)
        test_loss = test(model, test_data_loader, device)
        test_loss_arr.append(test_loss)
        print(f'Epoch: {epoch:02d}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss}')

        # Early stopping
        if min_loss < test_loss:
            epoch_without_improvement += 1
            if epoch_without_improvement >= patience:
                print("EARLY STOPPING")
                break
        else:
            epoch_without_improvement = 0
        min_loss = min(min_loss, test_loss)

    val_loss = test(model, val_data_loader, device)
    print(f'Validation Loss: {val_loss:.4f}')
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_directory, "model_weights.pth"))
    np.save(os.path.join(args.output_directory, "train_loss.npy"), np.array(train_loss_arr))
    np.save(os.path.join(args.output_directory, "test_loss.npy"), np.array(test_loss_arr))
    np.save(os.path.join(args.output_directory, "test_dataset.npy"), np.array(test_data))


if __name__ == '__main__':
    main()
