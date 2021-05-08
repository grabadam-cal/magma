import numpy as np
from itertools import combinations
import seaborn as sns
from scipy.spatial import distance as spdist
import tqdm
from torch_geometric.data import Data
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Batch
from rdkit import Chem


possible_atom_list = [
	'S', 'Si', 'F', 'Fl', 'O', 'C', 'I', 'P', 'Cl',
	'Br', 'N', 'Unknown'
]

possible_hybridization_list = [
			Chem.rdchem.HybridizationType.SP,
			Chem.rdchem.HybridizationType.SP2,
	        Chem.rdchem.HybridizationType.SP3,
	        Chem.rdchem.HybridizationType.SP3D,
	        Chem.rdchem.HybridizationType.SP3D2
		]


def atom_hot_encoding(atom, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if atom not in allowable_set:
        atom = allowable_set[-1]

    return list(map(lambda s: atom == s, allowable_set))

def safe_index(l, e):
	"Gets the index of elem e in list l."
	try:
		return l.index(e)

	# If not in list, map as unknown symbol's index
	except:
		return len(l)

def get_feature_list(atom):
	"Get features for a given atom using RDkit."

	features[safe_index(possible_atom_list, atom.GetSymbol())]
	return features

def bond_features(bond)->np.array:
	"""One hot encodes a single bond from a molecule."""

	bt = bond.GetBondType()

	bond_feats = [
		bt == Chem.rdchem.BondType.SINGLE,
		bt == Chem.rdchem.BondType.DOUBLE,
		bt == Chem.rdchem.BondType.TRIPLE,
		bt == Chem.rdchem.BondType.AROMATIC
	]

	return np.array(bond_feats).astype(np.float32)

def atom_features(atom, use_chirality = False)->np.array:

	"""
	Feature extraction for a single atom of a molecule.
	"""

	atom_feats = atom_hot_encoding(atom.GetSymbol(), possible_atom_list)

	atom_feats.append(atom.GetDegree())
	atom_feats.append(atom.GetFormalCharge())
	atom_feats.append(atom.GetNumRadicalElectrons())
	atom_feats.append(atom.GetIsAromatic())
	atom_feats.append(atom.GetImplicitValence())

	atom_feats.append(
		safe_index(possible_hybridization_list, atom.GetHybridization())
	)

	if use_chirality:
		try:
			atom_feats += atom_hot_encoding(atom.GetProp('_CIPCode'),['R', 'S'])

			atom_feats += [atom.HasProp('_ChiralityPossible')]

		except:
			atom_feats +=[False, False]
			atom_feats +=[atom.HasProp('_ChiralityPossible')]


	return np.array(atom_feats).astype(np.float32)

#TO-DO atom_features_pandas(mol):

def get_bond_pair(mol):#->tuple(list, np.array):
	"""
	Returns the indices and adjacency matrix for all bonds in a molecular graph.
	"""

	bonds = mol.GetBonds()
	indices =[[], []]
	n_atoms = mol.GetNumAtoms()
	adj = np.zeros((n_atoms, n_atoms))

	for bond in bonds:

		begin_ix = bond.GetBeginAtomIdx()
		end_ix = bond.GetEndAtomIdx()

		indices[0]+= [begin_ix]
		indices[1]+= [end_ix]


		adj[begin_ix, end_ix]=1
		adj[end_ix, begin_ix]=1

	return indices, adj

def mol2graph_data(mol)->tuple:
    """
    Returns node features, edge_indices, edge (bond features),
    and an adjacency matrix for a molecular graph.
    """
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    node_feats = [atom_features(atom) for atom in atoms]

    edge_ixs, adj = get_bond_pair(mol)

    edge_feats = [bond_features(bond) for bond in bonds]

    return np.stack(node_feats), np.stack(edge_ixs), np.stack(edge_feats)#, adj


def mol2tensors(mol):
    """
    Generates a torch_geometric.data.Data object from
    an RDkit molecule.
    """
    #node_feats, edge_ixs, edge_feats, adj = mol2graph_data(mol)
    node_feats, edge_ixs, edge_feats = mol2graph_data(mol)

    data = Data(
        x = torch.tensor(node_feats, dtype = torch.float),
        edge_index =torch.tensor(edge_ixs, dtype = torch.long),
        edge_attr = torch.tensor(edge_feats, dtype = torch.float)
    )

    return data


def n_atom_features():
	bond = Chem.MolFromSmiles('C').GetAtomWithIdx(0)

	return len(atom_features(atom))

def n_bond_features():
	bond = Checm.MolFromSmiles('CC'.GetBondWithIdx(0))
	return len(bond_features(bond))


def get_cam_weight(graph, model):
    """
    Assumes an architecture of a single fully-connected layer
    after the graph conv layers.

    Params
    ------
    graph(torch_geometric.Data.data.data)
        Graph to get node activations from.

    """
    # Get predicted class
    batch_ = Batch.from_data_list([graph])
    logits = model(batch_)
    top_ix = logits.argmax()

    # Get row of weight matrix corresponding
    # to predicted class
    weight_matrix_params = list(
        model._modules.get('final_layer').parameters()
    )

    weight_matrix = weight_matrix_params[0]

    w_top = weight_matrix[top_ix]

    return w_top

def get_gradCAM_activations(graph, model):
    """
    Returns the gradCAM activation scores per node in a graph.

    Params
    ------
    graph(torch_geometric.Data.data.data)
        Graph to get node activations from.

    model (nn.Module)
        Graph conv net classifier model.

    Returns
    -------
    grad_cam_avg (array-like)


    Notes: Grad CAM (https://arxiv.org/pdf/1610.02391.pdf) computes
    ∂(y_top) / ∂G, where G is the output of the last conv layer before
    average pooling, i.e. the last node embeddings.
    In this sense GradCAM is the linear approximation of the neural network
    downstream of G, and captures the importance of each node embedding for
    a target class.
    """

    model.eval()

    # Get graph embedding
    embedding = model.project(
        Batch.from_data_list([graph]),
        reg_hook = True,
        pool =True
    )

    # Make predictions without softmax
    logits = model.final_layer(embedding)

    top_ix = logits.argmax()

    #print('Predicted class: ', uniques[top_ix])

    # Call backward pass to compute gradients
    logits[0, top_ix].backward()

    # Extract gradients from hook
    gradients = model.get_activations_gradient()

    # Compute node embeddings
    with torch.no_grad():
        node_embeddings = model.project(
            Batch.from_data_list([graph]),
            pool = False,
            reg_hook=False
        )

    # Get grad CAM
    grad_cam = gradients*node_embeddings
    grad_cam_avg = grad_cam.mean(axis = 1).detach().numpy()

    return grad_cam_avg

def get_CAM_activations(graph, model, weight_vector):
    """
    Params
    ------
    graph(torch_geometric.Data.data.data)
        Graph to get node activations from.

    model (nn.Module)
        Graph conv net classifier model.

    weight_vector (torch.tensor)
        It is the row of the weight matrix that
        produced the highest logit `W[ix_top]`. This is precisely
        class activation map method.

    Returns
    ------
    node_activations (np.array)
        Array with Class Activation Map (CAM) scores per node.
    """

    x, edge_index = graph.x, graph.edge_index

    with torch.no_grad():

        # Run through conv layers
        for conv_layer in model.conv_encoder:
            x = conv_layer(x, edge_index)
            x = torch.tanh(x)

        #Compute node activation map scores
        act_map = weight_vector.view(1, -1) @ x.t()

    node_activations = act_map.numpy().flatten()

    return node_activations



def get_embedding_activations(graph, model, weight_matrix):
    """
    Returns node activation of a graph using the embedding activation
    method.

    Params
    ------
    graph(torch_geometric.Data.data.data)
        Graph to get node activations from.

    model (nn.Module)
        Graph conv network model. It must support node embedding
        generation using model.project(x, pool=False).

    weight_matrix (torch.tensor)
        Weight matrix, takes as input an average node embedding
        (i.e. graph embedding ϕ`) and converts into a lower dimensional
        graph embedding ϕ''.

    Returns
    -------
    node_activations(np.array)
        shape (nodes,)

    """

    with torch.no_grad():
        node_embeddings = model.project(graph, pool =False)
        node_activations = node_embeddings@weight_matrix.T

    return node_activations.numpy()


def plot_node_activations(
    mol,
    node_activations,
    fig_fname,
    plot_cbar = False
    ):

    """
    Saves a molecule colored by node activations in a directory
    specified by `fname`.

    Params
    ------
    mol (rdkit.Chem.rdchem.Mol)
        RDKit molecule.

    node_activations (array-like)
        Per node activation map generated by e.g. `get_CAM_activations()`.

    fig_fname(str)
        path to save file + file name

    plot_cbar(bool, default = False)
        Whether or not to plot colorbar to get a sense of scale.


    Notes: Uses viridis by default. Doesn't return an object.
    """

    min_act, max_act = node_activations.min(), node_activations.max()

    normalizer =mpl.colors.Normalize(vmin=min_act, vmax = max_act)

    cmap = cm.viridis

    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    d = rdMolDraw2D.MolDraw2DCairo(500, 500)

    n_atoms = len(list(mol.GetAtoms()))

    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        mol,
        highlightAtoms=list(range(n_atoms)),
        highlightAtomColors={
            i: mapper.to_rgba(node_activations[i]) for i in range(n_atoms)
        },
    )

    with open(fig_fname,'wb') as file:
        file.write(d.GetDrawingText())

    if plot_cbar:

        plt.imshow(
            np.linspace(min_act, max_act, 10).reshape(1, -1),
            cmap = cmap
        )
        plt.gca().set_visible(False)

        plt.colorbar(orientation = 'horizontal')
        plt.savefig(
            fig_fname.split('.png')[0] + '_cbar.png',
            dpi = 230,
            bbox_inches='tight'
        )

    return None

