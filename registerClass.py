import numpy as np
from qutip import *
from scipy.spatial.distance import pdist, squareform
from pulseClass import Pulse  # Add this import

class Register:
    def __init__(self, positions: np.ndarray, C6: float = 2*np.pi*8620):
        self.positions = np.asarray(positions)
        self.N = len(positions)
        self.C6 = C6
        
        # Basis states and projectors
        self.g = basis(2, 0)
        self.r = basis(2, 1)
        self.proj_r = self.r * self.r.dag()
        
        # Precompute pairwise distances and interactions
        self._compute_interactions()
        self._create_operators()
        
    def _compute_interactions(self):
        """Precompute interaction matrix between all atom pairs"""
        dist_matrix = squareform(pdist(self.positions))
        np.fill_diagonal(dist_matrix, np.inf)
        self.V_matrix = self.C6 / (dist_matrix**6)
        
    def _create_operators(self):
        """Initialize quantum operators"""
        # Single-atom operators
        self.sigma_rr = [self._single_atom_op(self.proj_r, i) for i in range(self.N)]
        trans_gr = self.g * self.r.dag()
        self.trans_gr = [self._single_atom_op(trans_gr, i) for i in range(self.N)]
        self.trans_rg = [self._single_atom_op(trans_gr.dag(), i) for i in range(self.N)]
        
        # Corrected interaction operators
        self.interaction_ops = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                op_list = [qeye(2) for _ in range(self.N)]
                op_list[i] = self.proj_r
                op_list[j] = self.proj_r
                op = tensor(op_list)
                self.interaction_ops.append((op, self.V_matrix[i,j]))

    def _single_atom_op(self, op, index: int):
        """Create operator acting on single atom"""
        return tensor([op if i == index else qeye(2) for i in range(self.N)])
    
    def get_observables(self):
        """Generate common observables for simulation"""
        obs = {}
        
        # Single-atom Rydberg populations
        for i in range(self.N):
            obs[f"P_r_{i}"] = self.sigma_rr[i]
        
        # Interaction energy terms
        for (i, j), V in self._pair_iterator():
            op_list = [qeye(2) for _ in range(self.N)]
            op_list[i] = self.proj_r
            op_list[j] = self.proj_r
            obs[f"V_{i}{j}"] = V * tensor(op_list)
            
        # Total Rydberg population
        obs["P_r_total"] = sum(obs[f"P_r_{i}"] for i in range(self.N))
        
        return obs

    def _pair_iterator(self):
        """Helper to iterate through atom pairs and their interactions"""
        for i in range(self.N):
            for j in range(i+1, self.N):
                yield (i, j), self.V_matrix[i,j]

    def build_hamiltonian(self, pulse):
        """
        Construct time-dependent Hamiltonian for the register
        Returns: (QobjEvo, args) tuple
        """
        if not hasattr(pulse, 'Omega') or not callable(pulse.Omega):
            raise TypeError("Input must be a valid Pulse object")

        H_terms = []
        args = {'pulse_params': pulse.parameters}

        # Rabi terms
        for i in range(self.N):
            H_rabi = (self.trans_gr[i] + self.trans_rg[i]) / 2
            H_terms.append([
                H_rabi, 
                lambda t, args: args['pulse_params']['Omega']['func'](t, args['pulse_params']) *
                            np.cos(args['pulse_params']['phi']['func'](t, args['pulse_params']))
            ])

        # Detuning terms
        for i in range(self.N):
            H_terms.append([
                -self.sigma_rr[i],
                lambda t, args: args['pulse_params']['delta']['func'](t, args['pulse_params'])
            ])

        # Interaction terms (time-independent)
        for op, strength in self.interaction_ops:
            H_terms.append(strength * op)

        return QobjEvo(H_terms, args=args), args

    def plot_configuration(self, pulse=None, ax=None):
        """Plot atomic positions with blockade visualization"""
        # Implementation from previous answer
        # ...

    def __repr__(self):
        return (f"RydbergRegister(N={self.N}, C6={self.C6/(2*np.pi):.1f}×2π GHz·µm⁶)\n"
                f"Positions:\n{self.positions}")