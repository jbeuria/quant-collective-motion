import numpy as cp
import numpy as np
from qutip import *

from qutip import *
from qutip.qip.operations import cphase, snot
import numpy as np

import networkx as nx
from itertools import combinations
from networkx.algorithms.isomorphism import GraphMatcher



def w_state(n):

    basis_states = [tensor(*[basis(2, 1) if i == j else basis(2, 0) for i in range(n)]) for j in range(n)]
    w_state = (1 / np.sqrt(n)) * sum(basis_states)
    # Compute the density matrix
    return w_state * w_state.dag()

def ghz_state(n_qubits):
    # |0> state for all qubits
    state_0 = tensor([basis(2, 0) for _ in range(n_qubits)])
    # |1> state for all qubits
    state_1 = tensor([basis(2, 1) for _ in range(n_qubits)])
    
    # Create the GHZ state
    ghz = (state_0 + state_1).unit()

    return ghz*ghz.dag()


def generate_connected_graphs(n):
    all_graphs = []
    seen_graphs = []
    all_edges=[]
    def is_unique(G):
        """Check if the graph G is unique up to isomorphism."""
        for seen_G in seen_graphs:
            matcher = GraphMatcher(G, seen_G)
            if matcher.is_isomorphic():
                return False
        return True

    # Generate all possible graphs
    possible_edges = list(combinations(range(n), 2))
    for r in range(n - 1, len(possible_edges) + 1):
        for edges in combinations(possible_edges, r):
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)
            if nx.is_connected(G) and is_unique(G):
                seen_graphs.append(G)
                all_graphs.append(G)
                all_edges.append(list(G.edges()))

    return all_edges


def get_cluster_states(n,dm=True):
    edges_list=generate_connected_graphs(n)
    plus = snot() * basis(2, 0)
    # Tensor product of n |+> states for the three qubits
    initial_state = tensor([plus for i in range(n)])
    cstates_list=[]
    for edges in edges_list:
        state=initial_state 
        for c,t in edges:
            state = cphase(np.pi, n, control=c, target=t) * state
        
        if(dm==True): cstates_list.append(state * state.dag())
        else: cstates_list.append(state)
    
    return cstates_list


def get_wstate_dm(n):

    basis_states = [tensor(*[basis(2, 1) if i == j else basis(2, 0) for i in range(n)]) for j in range(n)]
    w_state = (1 / np.sqrt(n)) * sum(basis_states)
    # Compute the density matrix
    return w_state * w_state.dag()

def ghz_state(n_qubits):
    # |0> state for all qubits
    state_0 = tensor([basis(2, 0) for _ in range(n_qubits)])
    # |1> state for all qubits
    state_1 = tensor([basis(2, 1) for _ in range(n_qubits)])
    
    # Create the GHZ state
    ghz = (state_0 + state_1).unit()

    return ghz*ghz.dag()

def bell_states():
    state_0 = basis(2, 0)
    state_1 = basis(2, 1)

    bell1=(1.0/np.sqrt(2)) *(tensor([state_0,state_0]) + tensor([state_1,state_1]))

    bell2=(1.0/np.sqrt(2)) *(tensor([state_0,state_0]) - tensor([state_1,state_1]))

    bell3=(1.0/np.sqrt(2)) *(tensor([state_0,state_1]) + tensor([state_1,state_0]))

    bell4=(1.0/np.sqrt(2)) *(tensor([state_0,state_1]) - tensor([state_1,state_0]))

    return [bell1 * bell1.dag(), bell2 * bell2.dag(), bell3 * bell3.dag(), bell4 * bell4.dag()]

def get_rho():
    

def get_neighbors_within_cone(positions, heading, alpha, rmin, rmax, box_size):
    """
    Find neighbors within a vision cone (solid angle for 3D) or vision cone for 2D, 
    with distances between `rmin` and `rmax` in a periodic box.

    Args:
        positions (np.ndarray): Nx2 or Nx3 array of x, y (and z for 3D) positions of N agents.
        alpha (float): Angle of the vision cone in radians (solid angle for 3D).
        rmin (float): Minimum distance to consider neighbors.
        rmax (float): Maximum distance to consider neighbors.
        heading (np.ndarray): Nx2 or Nx3 array of heading vectors for each agent.
        box_size (float): Size of the periodic box (assumed to be square in 2D or cubic in 3D).

    Returns:
        neighbors (list of np.ndarrays): Each entry is an array of neighbors for the corresponding agent.
    """
    N, dims = positions.shape  # Determine whether the input is 2D or 3D based on position shape
    
    # Compute direction vectors between all pairs of agents with periodic boundary conditions
    vec_ij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # Shape (N, N, dims)
    
    # Apply minimum image convention for periodic boundary conditions
    vec_ij = vec_ij - box_size * np.round(vec_ij / box_size)
    
    # Calculate the adjusted distance matrix based on the periodic positions
    adjusted_distances = np.linalg.norm(vec_ij, axis=2)  # Shape (N, N)
    
    # Distance filtering: Create a mask for distances within [rmin, rmax]
    valid_distances_mask = (adjusted_distances >= rmin) & (adjusted_distances <= rmax)
    
    # Normalize these vectors (direction vectors between agents)
    norms = np.linalg.norm(vec_ij, axis=2, keepdims=True)  # Shape (N, N, 1)
    normed_vec_ij = np.divide(vec_ij, norms, where=norms != 0)  # Avoid division by zero
    
    # Compute dot products between the heading of each agent and the vectors to others
    dot_products = np.einsum('ijk,ik->ij', normed_vec_ij, heading)  # Shape (N, N)
    
    # Calculate the angles from the dot products (using acos for both 2D and 3D)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))  # Shape (N, N), clamp dot products to avoid numerical issues
    
    # Angle filtering: Check if the angles are within the vision cone [-alpha/2, +alpha/2]
    valid_angles_mask = (angles <= alpha / 2)
    
    # Combine distance and angle filters
    valid_neighbors_mask = valid_distances_mask & valid_angles_mask
    
    # For each agent, extract the indices of valid neighbors
    neighbors = [np.where(valid_neighbors_mask[i])[0] for i in range(N)]
    
    return neighbors




def get_neighbours(pos,rmin,rmax,L,N):
    # del_r = cp.linalg.norm(points_gpu[:, None] - points_gpu, axis=-1)
    del_r=pos[:,None,:]-pos
    del_r = del_r - L*cp.rint(del_r/L ) #minimum image convention
    del_r2 = cp.sum(del_r**2,axis=-1)
    del_r=del_r2**0.5  

    neighbours = ((del_r <= rmax) & (del_r >= rmin)) # When rmin >0, this excludes the ith agent also
    
    return neighbours

def get_next_orientation(dmatrix,neighbours,ori,L,N,eta=0.2,nn=3,ep=0.8):

    # nn_count=cp.sum(neighbours, axis=1)
    # print(L,N,eta,nn,ep)
    # if(nn <=nn_count): n_qubits= nn 
    # else: n_qubits= nn_count
    new_ori_all=np.zeros_like(ori)
    OPs=[]
    nn_ids_all=[]
    for agent_idx in range(N):
        nn_ids = neighbours[agent_idx]
        nn_agent_ids = nn_ids
        n_qubits = len(nn_agent_ids)
       
        if(len(nn_ids) > nn): 
            nn_agent_ids = np.random.choice(nn_ids, size=nn, replace=False)
            n_qubits= nn

        nn_ids_all.append(nn_agent_ids)    
        # print(nn_ids,nn_agent_ids,nn)
        # dmatrix=  w_dm #generate_random_density_matrix(n_qubits, ep)
        d=2**n_qubits # dimension of density matrix

        labels=[np.binary_repr(i).zfill(n_qubits)  for i in np.arange(0,d)]
        # nonzero_indices=[[i for i, char in enumerate(label) if char != '0'] for label in labels]
 
        masks=[np.fromiter(label, dtype=int) for label in labels]

        op=[np.zeros_like(dmatrix.full()) for i in range(len(L))] # one operator for each dimension being used.
        ori_nns=ori[nn_agent_ids,:]

        # print(ori_nns.shape,len(masks[0]))
        ori_agent=[]
        
        # print(ori_agent)
        # print("************ ori ************")

        op_dims=[2 for i in range(nn)]
        
        # print(op_dims)
        for dim in range(len(L)): # for each component of operator
            for i in range(d):
                for j in range(d):
                    if(i != j): op[dim][i,j]= (masks[i].dot(ori_nns[:,dim]) + masks[j].dot(ori_nns[:,dim]))

            # print(op[dim])
           
            op_qobj=Qobj(op[dim],dims=[op_dims,op_dims])
            
            OPs.append(op)
            # print(dmatrix)
            # print(op_qobj)

            e_dim=np.real((dmatrix * op_qobj).tr())
            # if(np.isnan(e_dim)): print(op_qobj)
            # print(n_qubits,e_dim)
            ori_agent.append(e_dim + ori[agent_idx,dim])
            # print(n_qubits,ori_agent[dim])
        
        # print(ori_agent)
        # print(np.linalg.norm(ori_agent))

        # ori_agent=np.asarray(ori_agent)
        # print(ori_agent)
        ori_agent=ori_agent/np.linalg.norm(ori_agent)

        # if NAN found, revert to old direction
        # ori_agent=np.asarray(ori_agent)
        
        # if NAN found, revert to old direction
        if(np.isnan(ori_agent).any()): 
            ori_agent = ori[agent_idx,:]
        
        new_ori_all[agent_idx]=ori_agent
        
    noise=cp.random.normal(scale=eta, size=(N, len(L)))
    ori_new_nosiy= new_ori_all +noise

    ori_new_nosiy= ori_new_nosiy/(cp.sqrt(cp.sum(ori_new_nosiy**2,axis=1))[:,None])
    return ori_new_nosiy,nn_ids_all


def vicsek_ricci(dmatrix,filename,v=0.3,rmin=0,rmax=1.0,nn=3, 
                         eta=0.3, N=150,n_steps=3000, 
                         L=[100,100],dt=0.1,alpha=np.pi/2):
    
    L = cp.asarray(L)
    pos = L*cp.random.rand(N,len(L))
    ori = cp.random.rand(N, len(L))-0.5
    ori= ori/(cp.sqrt(cp.sum(ori**2,axis=1))[:,cp.newaxis]) # normalize the orientation
    pos_history=[]
    ori_history=[]
    NN_history=[]
    for i in range(n_steps+1):
        if(i%100==0): print(f"************************ step: {i}")
        # neighbours = get_neighbours(pos,rmin,rmax,L,N)
        neighbours = get_neighbors_within_cone(pos, ori, alpha, rmin, rmax, L[0])
        # to run on GPU with cupy
        # pos_history.append(pos.get())
        # ori_history.append(ori.get())

        # to run on CPU
        pos_history.append(pos)
        ori_history.append(ori)

        pos =pos+ dt*(v*ori)
        pos =pos -L*cp.floor(pos /L)
        ori,nn_all = get_next_orientation(dmatrix,neighbours,ori,L,N,eta,nn,ep=0.95)
        NN_history.append(nn_all)
        if(i%100==0): 
            open('log-2d.txt', "w").write(f"step: {i}\n")
        
        # if((i+1)%1000==0):
        #     counter=int(i/1000)

    f=f"{filename}"
    np.save(f,{'X':pos_history ,'V':ori_history,'NN':NN_history})
    
        # pos_history=[]
        # ori_history=[]
            
    return (pos_history,ori_history)

def phase_trans(eta):
        labels=[]
        # if(nn==1): continue
        dmat_list=bell_states() + [w_state(3),ghz_state(3)] + get_cluster_states(2) + get_cluster_states(3) + get_cluster_states(4)
        labels=['bell1','bell2','bell3','bell4','w_state']
        labels +=['ghz_state',
                'cluster2_1','cluster3_1','cluster3_2',
                'cluster4_1','cluster4_2','cluster4_3','cluster4_4','cluster4_5','cluster4_6']
        for idx,dmatrix in enumerate(dmat_list):
            # print(dmatrix)
            nn=len(dmatrix.dims[0])
            label=labels[idx]
            eta=np.round(eta,1)
            filename=f"output_alpha180_rmin0.1_rmax5.0/traj_nn{nn}_{label}_eta{eta}"
            X,V=vicsek_ricci(dmatrix,filename,v=0.5,rmin=0.1,rmax=5.0,nn=nn,eta=eta, 
                            N=200, n_steps=1000, L=[10,10],alpha=np.pi)

from joblib import Parallel, delayed

Parallel(n_jobs=-1)(delayed(phase_trans)(eta) for eta in np.linspace(0,2.0,21))

