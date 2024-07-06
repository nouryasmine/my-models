import base64
from io import BytesIO
from typing import List
from fastapi import HTTPException, Response
import torch
from pathlib import Path
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Compose, NormalizeFeatures
from model.utils import load_from_file, sample
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import Draw
import os 

BASE_DIR = Path(__file__).resolve(strict=True).parent
script_dir = os.path.dirname(__file__)



class GAT(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 128, heads=8, dropout=0.7)
        self.conv2 = GATConv(128 * 8, 64, heads=8, dropout=0.7)
        self.conv3 = GATConv(64 * 8, 32, heads=4, dropout=0.7)
        self.out = torch.nn.Linear(32 * 4, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.7, training=self.training)
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.7, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.7, training=self.training)
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = self.out(x)
        return x

num_node_features = 11  # Number of features per node in the QM9 dataset
model = GAT(num_node_features)

model = model.to('cpu')

#with open (f"{BASE_DIR}/mymodel.pth","rb") as f:
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model_state.pth')))
model.eval()

model2=  GAT(num_node_features)
model2.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model_state_homo.pth')))
model2.eval()

model_path = os.path.join(BASE_DIR, 'pretrained.rnn.pth')
pretrained_rnn_model = load_from_file(model_path)

model3=  GAT(num_node_features)
model3.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model_state_dipolemoment.pth')))
model3.eval()

model4=  GAT(num_node_features)
model4.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model_state_lumo.pth')))
model4.eval()

model5=  GAT(num_node_features)
model5.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model_state_zpve.pth')))
model5.eval()


def atom_features(atom):
    features = [
        float(atom.GetAtomicNum()),                     # Atomic number
        float(atom.GetChiralTag()),                     # Chirality (as a float)
        float(atom.GetDegree()),                        # Degree (number of bonds)
        float(atom.GetFormalCharge()),                  # Formal charge
        float(atom.GetTotalNumHs()),                    # Number of bonded hydrogen atoms
        float(atom.GetHybridization()),                 # Hybridization state (as float)
        float(atom.GetIsAromatic()),                    # Aromaticity (as float)
        float(atom.GetTotalNumHs()),                    # Total number of implicit hydrogens
        float(atom.GetTotalDegree()),                   # Total degree
        float(atom.GetImplicitValence()),               # Implicit valence
        float(atom.GetExplicitValence())                # Explicit valence
    ]
    return features

# Convert a SMILES string to a PyTorch Geometric Data object
def smiles_to_graph(smiles : str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Could not parse SMILES string: {smiles}")
        return None



    # Get adjacency matrix
    adj = GetAdjacencyMatrix(mol)
    
    # Get atom features
    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
    
    # Create edge index
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    
    if edge_index.size(1) == 0:
        print(f"Error: No edges found in the adjacency matrix for SMILES string: {smiles}")
        return None

    
    # Create node features
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    
    # Create data object
    data = Data(x=x, edge_index=edge_index)
    return data


def convert_to_smiles(molecules):
    smiles_list = []
    for mol in molecules:
        smiles = Chem.MolToSmiles(mol)
        smiles_list.append(smiles)
    return smiles_list

def predict_gap(molecule : str )-> float:
    m=smiles_to_graph(molecule)
    pred= model(m)
    return pred.item()


def predict_homo(molecule : str )-> float:
    m=smiles_to_graph(molecule)
    pred1= model2(m)
    return pred1.item()

def predict_lumo(molecule : str )-> float:
    m=smiles_to_graph(molecule)
    pred3= model4(m)
    return pred3.item()


def predict_dipolemoment(molecule : str )-> float:
    m=smiles_to_graph(molecule)
    pred2= model3(m)
    return pred2.item()

def predict_zpve(molecule : str )-> float:
    m=smiles_to_graph(molecule)
    pred4= model5(m)
    return pred4.item()


def generate_molecules():
    generated_molecules = []
    while len(generated_molecules) != 10:
        # Generate token sequences
        sequences, nlls = sample(model=pretrained_rnn_model)
        
        # Convert the token sequences into SMILES
        smiles = pretrained_rnn_model.tokenizer.untokenize(pretrained_rnn_model.vocabulary.decode(sequences[0].numpy()))
        
        # Transform the generated SMILES into RDKit Mol objects
        # The Mol object is "None" if the SMILES cannot be parsed by RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # At this point, the Mol is valid so let's keep track of it
            generated_molecules.append(mol)
        smiles_list= convert_to_smiles(generated_molecules)
    return smiles_list

def visualize_molecules(smile_list: List[str]):
    generated_mols = [Chem.MolFromSmiles(smiles) for smiles in smile_list]
    img = Draw.MolsToGridImage(generated_mols, molsPerRow=5)
    buf = BytesIO()
    img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return byte_im

def smilesToMol(smiles_list: List[str]):
    processed_molecules = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        processed_molecules.append(mol)
    return processed_molecules