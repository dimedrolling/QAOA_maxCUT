import numpy as np
import itertools
import qiskit.quantum_info as qi 
import time
import time
import csv
import sys
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
from functools import reduce
from scipy.linalg import expm





# produce all binary strings of length n with k 1s. If k is None then all possible binary strings of length n produced
def get_binary_strings(n, k=None) -> list:
    '''
    produce all binary strings of length n with k 1s
    returns list with binary lists 
    '''
    final = []
    def kbits(r):
        result = []
        for bits in itertools.combinations(range(n), r):
            s = [0] * n
            for bit in bits:
                s[bit] = 1
            result.append(s)   
        return result

    if k != None:
        return kbits(k)
    
    for i in range(n + 1):
        final = final + kbits(i)
        
    return final

def get_circuit_operators( lst_x=None, lst_z=None):
    '''
    note the order! (x,z)
    '''
    return  qi.Pauli((lst_z,lst_x)).to_matrix()




def generate_QAOA_operator(nqubits, locality=2, number_of_terms=10, number_of_layers = 1 , beta_angle = [], gamma_angle = []):
    """_summary_

    Args:
        nqubits (_type_): _description_
        locality (_type_): _description_
        number_of_terms (_type_): _description_
        number_of_layers (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """    '''

    '''

    # define angles for mixers 
    if len(beta_angle)==0:
        beta_angle = np.random.rand(number_of_layers) * np.pi/3
    cos_angle_beta = np.ones(number_of_layers)*np.cos(beta_angle)
    sin_angle_beta = np.ones(number_of_layers)*np.sin(beta_angle)

    if len(gamma_angle)==0:
        gamma_angle = np.random.rand(number_of_layers) * np.pi/6
    cos_angle_gamma = np.ones(number_of_layers)*np.cos(gamma_angle)
    sin_angle_gamma = np.ones(number_of_layers)*np.sin(gamma_angle)
 

    # making Hamiltonian 

    locality = min(locality, nqubits)
    number_of_terms = min(number_of_terms,nqubits)

    # Hamiltonian  ZjZk
    rng = np.random.default_rng()
    Zs = get_binary_strings(nqubits,locality)
    # pickedZ = np.random.choice(Zs,size=number_of_terms,replace=False)
    pickedZ = rng.choice(Zs, number_of_terms, replace=False)

    # Making Hamiltonian
    H = 0
    for i in range(0,len(pickedZ)):
        H = H + get_circuit_operators(np.zeros(nqubits),pickedZ[i])

    ans = H

    for p in range(number_of_layers):
        # Making unitary for Hamiltonian exp
        identity = get_circuit_operators(np.zeros(nqubits),np.zeros(nqubits))

        
        unitary_z = identity

        for i in range(0,len(pickedZ)):
            unitary_z = unitary_z@(cos_angle_beta[p]*identity \
            - 1j*sin_angle_beta[p]*get_circuit_operators(np.zeros(nqubits),pickedZ[i]))
            # unitary_z = unitary_z @ np.exp(-1j*beta_angle[p]*get_circuit_operators(np.zeros(nqubits),pickedZ[i]))


        # Making mixer X
        x_string = np.zeros(nqubits)

        unitary_x = identity

        for i in range(0,nqubits):
            x_string[i] = 1
            unitary_x = unitary_x@(cos_angle_gamma[p]*identity \
            - 1j*sin_angle_gamma[p]*get_circuit_operators(x_string,np.zeros(nqubits)))
            x_string[i] = 0


        ans = unitary_z @ unitary_x @ ans @ unitary_x.conjugate() @ unitary_z.conjugate()

    return ans
   
   




def count_solutions_XI( Operator: np.array, tol=1e-10 ) -> tuple:
    """_summary_

    Args:
        nqubits (int): _description_
        Operator (np.array): _description_
        tol (_type_, optional): _description_. Defaults to 1e-10.

    Returns:
        tuple: _description_
    """    '''
    '''
    nqubits = int(np.log2(len(Operator)))
    x_strings = get_binary_strings(nqubits)
    z_zeros = np.zeros(nqubits)

    ans = []
    max_locality  = 0
    avg_locality = 0.0

    for i in x_strings:
        x_mat = get_circuit_operators(i,z_zeros)
        coef = np.around( 1/(1<<nqubits) * np.real(np.trace(x_mat@Operator)) ,6)
        if np.abs(coef)>tol:
            count_x = np.sum(i)
            ans.append((str(i),coef))
            if count_x > max_locality:
                max_locality = count_x
            avg_locality += count_x
    len_ans = len(ans)
    if len_ans == 0:
        len_ans = 1
    return ans, max_locality, avg_locality/len_ans, len_ans



def make_grid(nqubits: int, max_loc: int, max_terms: int, number_of_layers=1, beta_angle=[], gamma_angle=[])->np.ndarray:
    grid = np.zeros((max_loc,max_terms,3))
    for i in range(1,max_loc+1):
        for j in range(1,max_terms+1):
            if i==nqubits and j>1:
                continue
            ans = generate_QAOA_operator(nqubits=nqubits,locality=i,number_of_terms=j, number_of_layers=number_of_layers,beta_angle=beta_angle,gamma_angle=gamma_angle)
            ans, max_loc, avg_loc, len_ans = count_solutions_XI(ans,1e-10)
            grid[i-1][j-1][0] = max_loc
            grid[i-1][j-1][1] = avg_loc
            grid[i-1][j-1][2] = len_ans                
    return grid

def plot_grid(grid,layer=-1, mode=2):

    localities = [i + 1 for i in range(grid.shape[0])]
    num_of_terms = [i + 1 for i in range(grid.shape[1])]

    param_to_show_name = ''
    if mode == 2:
        param_to_show_name = 'total terms'
    elif mode == 1:
        param_to_show_name = 'average locality'
    else:
        param_to_show_name = 'max locality'

    fig, ax = plt.subplots()

    im = ax.imshow(grid[:,:,mode], interpolation='nearest', cmap='plasma')


    ax.set_xticks(np.arange(len(localities)), labels=localities)
    ax.set_yticks(np.arange(len(num_of_terms)), labels=num_of_terms)
    
    for locality in localities:
        for number_of_terms in num_of_terms:
            ax.text(number_of_terms-1, locality-1, '{:.2f}'.format(grid[locality-1][number_of_terms-1][mode]),  ha='center', va='center',color='black',fontsize = 15,backgroundcolor='w')
            
    ax.set_xlabel('No. of Terms', fontsize = 15)
    ax.set_ylabel('Locality', fontsize = 15)
    if layer==-1:
        ax.set_title('{} qubits, {}'.format(len(localities), param_to_show_name), fontsize = 20)
    else:
        ax.set_title('{} qubits {} layers, {}'.format(len(localities), layer, param_to_show_name), fontsize = 20)
    fig.tight_layout()
    plt.show()




def make_H_maxCUT(G, full_matrix=False):   
    if type(G) == nx.classes.graph.Graph:
            n = len(G.nodes())
            adj_mat = np.zeros([n, n])
            for i in range(n):
                for j in range(n):
                    temp = G.get_edge_data(i, j, default=0)
                    if temp != 0:
                        adj_mat[i, j] = 1
    else:
        adj_mat = G
    n = (len(adj_mat))

    H = np.zeros((2**n,2**n),dtype=complex)
    zeros = np.zeros(n)

    for i in range(n):
        for j in range(i+1,n):
            if adj_mat[i][j]!=0:
                buf = np.zeros(n)
                buf[i] = buf[j] = 1
                H += adj_mat[i][j]*get_circuit_operators(zeros, buf)
    return H


PAULIS = {'I': np.eye(2),
          'X': np.array([[0, 1], [1, 0]]),
          'Y': np.array([[0, -1j], [1j, 0]]),
          'Z': np.diag([1, -1])}

def get_pauli_decomp(in_matrix, display_every=5000, PAULIS=PAULIS):
    """Return dictionary with pauli strings and according weight.
    in_matrix - input matrix (should be of size(2^m,2^m), where m is int.
    display_time - if given int number print time spend for computation if False do not print.
    PAULIS - dictionary with Pauli basis.

    """
    m = int(np.log2(in_matrix.shape[0]))

    pauli_weights = {}

    k = 1
    K = len(list(product(PAULIS.keys(), repeat=m)))

    start_ = time.time()
    for u in product(PAULIS.keys(), repeat=m):
        pauli_str_name = ''.join(u)
        pauli_str_matrix = reduce(np.kron, [PAULIS[s] for s in u])
        inner_product = np.trace(in_matrix @ pauli_str_matrix) / 2**m
        if not np.isclose(inner_product, 0):
            pauli_weights[pauli_str_name] = inner_product
        if display_every and k%display_every == 0:
            print('\t {} qubits || {}/{} || {:.3f} s passed'.format(m,k,K, time.time()-start_))
        k+=1

    return pauli_weights



def construct_QAOA_operator_from_H(H, number_of_layers = 1 , beta_angle = [], gamma_angle = []):
    """_summary_

    Args:
        H (_type_): _description_
        number_of_layers (int, optional): _description_. Defaults to 1.
        beta_angle (list, optional): _description_. Defaults to [].
        gamma_angle (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """    '''
    '''
    # define angles for mixers 
    if len(H.shape)==1:
        H = np.diag(H)
    if len(beta_angle)==0:
        beta_angle = np.random.rand(number_of_layers) * np.pi/3


    if len(gamma_angle)==0:
        gamma_angle = np.random.rand(number_of_layers) * np.pi/6

    ans = np.copy(H)
    nqubits = int(np.log2(len(H)))

    mixer = 0
    x_string = np.zeros(nqubits)

    for i in range(0,nqubits):
        x_string[i] = 1
        mixer = mixer + get_circuit_operators(x_string, np.zeros(nqubits))
        x_string[i] = 0

    for p in range(number_of_layers):
        # Making unitary for Hamiltonian exp
        exp_H = expm(1j*H*beta_angle[p])
        # Making mixer X
        exp_X = expm(1j*mixer*gamma_angle[p])

        ans = exp_H @ exp_X @ ans @ exp_X.T.conjugate() @ exp_H.T.conjugate()

    return ans



def tensor(k):
    t = k[0]
    i = 1
    
    while i < len(k) :
        t = np.kron(t,k[i])
        i+=1
    return t


def Graph_to_Hamiltonian(G,n): 
    H = np.zeros((2**n), dtype = 'float64') 
    Z = np.array([1,-1],dtype = 'float64')
    
    for i in range(n):
        j = i+1
        while j<n:
            k = [[1,1]]*n
            k = np.array(k,dtype = 'float64')
                
            if G[i][j] !=0:
                k[i] = Z
                k[j] = Z
                H+= tensor(k)*G[i][j]
                    
            j+=1
        
    return H
def get_Hamiltonian(G):
    graph_array = nx.to_numpy_array(G)
    
    H = Graph_to_Hamiltonian(graph_array, G.number_of_nodes())
    
    return H
    