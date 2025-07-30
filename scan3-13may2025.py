import numpy as np
from qutip import Qobj
import numpy as cp
import numpy as np
from qutip import *
import os
from qutip import *
from qutip.qip.operations import cphase, snot
import numpy as np

import networkx as nx
from itertools import combinations
from networkx.algorithms.isomorphism import GraphMatcher
from joblib import Parallel, delayed


def get_n_puredm(n,sup=False):
    # Step 1: Generate 8 positive random coefficients (for 3 qubits: 2^3 = 8 basis states)
    
    if(sup): coeffs = np.ones(2**n)
    else: coeffs = np.random.rand(2**n)  # Random positive numbers
    
    # Step 2: Normalize the coefficients to form a valid quantum state
    coeffs /= np.linalg.norm(coeffs)
    # Step 3: Create the state vector |ψ⟩
    # Basis ordering: |000>, |001>, |010>, ..., |111>
    psi_vector = Qobj(coeffs.reshape(2**n, 1), dims=[[2]*n, [1]*n])
    # Step 4: Create the density matrix ρ = |ψ⟩⟨ψ|
    rho = psi_vector * psi_vector.dag()
    return rho

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

def bell_states():
    state_0 = basis(2, 0)
    state_1 = basis(2, 1)

    bell1=(1.0/np.sqrt(2)) *(tensor([state_0,state_0]) + tensor([state_1,state_1]))

    bell2=(1.0/np.sqrt(2)) *(tensor([state_0,state_0]) - tensor([state_1,state_1]))

    bell3=(1.0/np.sqrt(2)) *(tensor([state_0,state_1]) + tensor([state_1,state_0]))

    bell4=(1.0/np.sqrt(2)) *(tensor([state_0,state_1]) - tensor([state_1,state_0]))

    return [bell1 * bell1.dag(), bell2 * bell2.dag(), bell3 * bell3.dag(), bell4 * bell4.dag()]


rhos=bell_states() + [ghz_state(3) , w_state(3)]
labels=['bell1','bell2','bell3','bell4','ghz','w']

def log_to_file(message, log_file="log.txt"):
    print(message + "\n")
    with open(log_file, "a") as f:  # 'a' mode appends to the file
        f.write(message + "\n")

def find_X(O, dO_dt):
    try:
        O += np.eye(O.shape[0]) * 1e-10
        dO_dt += np.eye(dO_dt.shape[0]) * 1e-10
        
        n = O.shape[0]
        # Create the augmented matrix for the system of equations
        A = np.kron(np.eye(n), O) - np.kron(O.T, np.eye(n))  # Kronecker product for the equation OX - XO = D
        b = dO_dt.flatten()  # Flatten dO/dt to match dimensions    
        Ainv=np.asarray(np.linalg.pinv(A))
        X_flat=Ainv@b
        X = X_flat.reshape(n, n)
        return X
    except Exception as e:
        error_message = str(e)
        log_to_file(error_message,"error.txt")
        return np.zeros(O.shape)
        
def get_Pmax(OP):
    P_all=[]
    for agent_idx in range(len(OP)):
            P=[]
            for op_dim in OP[agent_idx]:
                P_dim=np.linalg.eigvals(op_dim)
                P.append(np.max(P_dim))
            P_all.append(P)
    return np.linalg.norm(np.sum(P_all,axis=0))                  

def get_Emax(OP1,OP2):
    E_all=[]
    for agent_idx in range(len(OP1)):
            E=[]
            for idx, op1_dim in enumerate(OP1[agent_idx]):
                op2_dim=OP2[agent_idx][idx]
                H=find_X(op1_dim, op2_dim-op1_dim)
                E_dim=np.linalg.eigvals(1.0j*H)
                E.append(np.max(E_dim))

            E_all.append(np.sum(E))
    return np.sum(E_all)



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

def get_next_orientation(dmatrix,neighbours,ori,L,N,eta=0.2,nn=3,v=0.5,ep=0.8):

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

        unit_vec = np.random.normal(size=3); 
        unit_vec /= np.linalg.norm(unit_vec)

        # print(op_dims)
        delta_ori=[]
        for dim in range(len(L)): # for each component of operator
            for i in range(d):
                for j in range(d):
                    if(i != j): op[dim][i,j]= (v/nn)*(masks[i].dot(ori_nns[:,dim]) + masks[j].dot(ori_nns[:,dim]))
                    else: 
                        op[dim][i,j]=eta*unit_vec[dim]
           
            op_qobj=Qobj(op[dim],dims=[op_dims,op_dims])
            OPs.append(op)
            e_dim=np.real((dmatrix * op_qobj).tr())
            delta_ori.append(e_dim)
            
        # print(ori[agent_idx,:],delta_ori)
        ori_agent=ori[agent_idx,:]+ np.asarray(delta_ori)/np.linalg.norm(delta_ori)
        
        ori_agent=ori_agent/np.linalg.norm(ori_agent)

        # if NAN found, revert to old direction
        # ori_agent=np.asarray(ori_agent)
        
        # if NAN found, revert to old direction
        if(np.isnan(ori_agent).any()): 
            ori_agent = ori[agent_idx,:]
        
        new_ori_all[agent_idx]=ori_agent
        
    # noise=cp.random.normal(scale=eta, size=(N, len(L)))
    # ori_new_nosiy= new_ori_all  +noise
    ori_new_nosiy= new_ori_all 

    ori_new_nosiy= ori_new_nosiy/(cp.sqrt(cp.sum(ori_new_nosiy**2,axis=1))[:,None])
    return ori_new_nosiy,nn_ids_all,OPs


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
    OP_history=[]
    E_history=[]
    P_history=[]

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

        dm=dmatrix
        
        ori,nn_all,ops = get_next_orientation(dm,neighbours,ori,L,N,eta,nn,v,ep=0.95)
        NN_history.append(nn_all)
        OP_history.append(ops)

        if(len(OP_history)>=2):
            Emax=get_Emax(OP_history[-2],OP_history[-1])
            E_history.append(Emax)

        Pmax=get_Pmax(ops)
        P_history.append(Pmax)
        
        if(i%100==0): 
            open('log-2d.txt', "w").write(f"step: {i}\n")
        
        # if((i+1)%1000==0):
        #     counter=int(i/1000)

    f=f"{filename}"
    np.save(f,{'X':pos_history ,'V':ori_history,'P':P_history,'E':E_history})
    
        # pos_history=[]
        # ori_history=[]
            
    return (pos_history,ori_history,P_history,E_history)


etas=np.linspace(0,2.0,21)
n_samples=10
markers=[]
labels=['random_2','random_3','sup_2','sup_3']
for idx,label in enumerate(labels):
    for eta in etas:
        for sample in range(n_samples):
            markers.append([idx,label,eta,sample])

def run_simulation(marker,rhos,directory="trajs_alpha90"):
    idx,label,eta,sample=marker    
    print(f"label: {label}, eta: {eta} ***********************\n")

    if('random' in label):
        nn=int(label.split('_')[1])
        dmatrix=get_n_puredm(nn,sup=False)
    elif('sup' in label):
        nn=int(label.split('_')[1])
        dmatrix=get_n_puredm(nn,sup=True)
        
    eta=np.round(eta,1)
    
    if not os.path.exists(directory): os.makedirs(directory)
    
    filename=f"{directory}/traj_nn{nn}_{label}_eta{eta}_{sample}"
    X,V,P,E=vicsek_ricci(dmatrix,filename,v=0.5,rmin=0.1,rmax=5.0,nn=nn,eta=eta, 
                    N=200, n_steps=250, L=[10,10],alpha=0.5*np.pi)


Parallel(n_jobs=-1)(delayed(run_simulation)(marker,rhos,"trajs_alpha90") for marker in markers)
      