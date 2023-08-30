import numpy as np
import itertools
import qiskit.quantum_info as qi 
import time
import time
import csv
import sys



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




def generate_QAOA_operator(nqubits, locality, number_of_terms, number_of_layers = 1 , beta_angle = [], gamma_angle = []):
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


        # Making mixer X
        x_string = np.zeros(nqubits)
        x_string[0] = 1
        unitary_x = identity

        x_string[0] = 0
        for i in range(0,nqubits):
            x_string[i] = 1
            unitary_x = unitary_x@(cos_angle_gamma[p]*identity \
            - 1j*sin_angle_gamma[p]*get_circuit_operators(x_string,np.zeros(nqubits)))
            x_string[i] = 0


        ans = unitary_z @ unitary_x @ ans @ unitary_x.conjugate() @ unitary_z.conjugate()

    return ans
   
   




def count_solutions_XI(nqubits: int, Operator: np.array, tol=1e-10 ) -> tuple:
    """_summary_

    Args:
        nqubits (int): _description_
        Operator (np.array): _description_
        tol (_type_, optional): _description_. Defaults to 1e-10.

    Returns:
        tuple: _description_
    """    '''
    '''
    x_strings = get_binary_strings(nqubits)
    z_zeros = np.zeros(nqubits)

    ans = []
    max_locality  = 0
    avg_locality = 0.0

    for i in x_strings:
        x_mat = get_circuit_operators(i,z_zeros)
        coef = 1/(1<<nqubits) * np.trace(x_mat@Operator)
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
            ans, max_loc, avg_loc, len_ans = count_solutions_XI(nqubits,ans,1e-10)
            grid[i-1][j-1][0] = max_loc
            grid[i-1][j-1][1] = avg_loc
            grid[i-1][j-1][2] = len_ans                
    return grid



def file_dump(line, name, path):
    with open(path + name, 'a') as f:  #####   ADD HERE PATH
        w = csv.writer(f, delimiter=',',lineterminator='\n')
        w.writerow(line)

def file_dump_np(nparr, name, path):
    with open(path + name, 'a') as f:  #####   ADD HERE PATH
        # f.write("\n")
        np.savetxt(f, nparr, delimiter=',',newline='\n',fmt='%1.3f')

def numpy_array_to_text(a):
    row = []
    for i in range(len(a)):
        row.append(np.real(a[i]))
        row.append(np.imag(a[i]))
    return row

def new_file_dump(line, name,path):
    with open(path + name, 'w') as f:
        w = csv.writer(f, delimiter=',',lineterminator='\n')
        w.writerow(line)


def compiler(n_qubits, n_layers, n_terms, n_locality):

    t0 = time.time()
    beta_angle = np.around(np.random.rand(n_layers) * np.pi/3, decimals=5)
    gamma_angle = np.around(np.random.rand(n_layers) * np.pi/6, decimals=5)
    grid = make_grid(n_qubits, max_loc=n_locality, max_terms=n_terms,number_of_layers=n_layers,beta_angle=beta_angle,gamma_angle=gamma_angle)   
    tf = time.time() - t0

    return grid, tf, beta_angle, gamma_angle



n_qubits = int(sys.argv[1])
n_layers = int(sys.argv[2])
n_terms = int(sys.argv[3])
n_locality = int(sys.argv[4])
path = sys.argv[5]
ind = int(sys.argv[6])

seed = int(np.mod(ind, 200))
np.random.seed(seed)


grid, tf, beta_angle, gamma_angle = compiler(n_qubits,n_layers,n_terms,n_locality)


new_file_dump([n_qubits, n_layers, n_terms, n_locality],
          f'{n_qubits}n_{n_layers}p_{n_terms}t_{n_locality}l_{seed}seed.csv', path)
file_dump([tf], f'{n_qubits}n_{n_layers}p_{n_terms}t_{n_locality}l_{seed}seed.csv', path)
file_dump(beta_angle, f'{n_qubits}n_{n_layers}p_{n_terms}t_{n_locality}l_{seed}seed.csv', path)
file_dump(gamma_angle, f'{n_qubits}n_{n_layers}p_{n_terms}t_{n_locality}l_{seed}seed.csv', path)
file_dump_np(grid.reshape(grid.shape[0], -1), f'{n_qubits}n_{n_layers}p_{n_terms}t_{n_locality}l_{seed}seed.csv', path)



