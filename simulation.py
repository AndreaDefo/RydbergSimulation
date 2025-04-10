from pulseClass import Pulse
from registerClass import Register
import numpy as np
import matplotlib.pyplot as plt
from qutip import *




# Import your custom classes (Pulse and Register) here
# ... [Include the Pulse and Register class definitions from previous steps] ...

# ======================
# Simulation Parameters
# ======================
C6 = 2 * np.pi * 8620  # MHz·µm⁶ (Rb 70S state)
R = 3.0  # µm (blockade radius ~3.7 µm for Ω=1 MHz)
Omega = 2 * np.pi * 1.0  # MHz (Rabi frequency)
t_max = 5  # µs (simulation duration)

# ======================
# System Configuration
# ======================
# Create atomic register
positions = np.array([[0.0, 0.0, 0.0], [R, 0.0, 0.0]])
register = Register(positions, C6=C6)

# Create constant pulse
pulse = Pulse()
pulse.set_parameter('Omega', 'constant', value=Omega)
pulse.set_parameter('delta', 'constant', value=0.0)
pulse.set_parameter('phi', 'constant', value=0.0)

# ======================
# Hamiltonian Construction
# ======================
H, ham_args = register.build_hamiltonian(pulse)

# ======================
# Time Evolution
# ======================

# Define all possible basis states
basis_labels = ['gg', 'gr', 'rg', 'rr']
basis_states = [
    tensor(register.g, register.g),
    tensor(register.g, register.r),
    tensor(register.r, register.g),
    tensor(register.r, register.r)
]

# ======================
# Time Evolution
# ======================
tlist = np.linspace(0, t_max, 100)
psi0 = tensor(register.g, register.g)  # Initial state |gg⟩

# Run simulation and save full state vectors
result = mesolve(H, psi0, tlist, e_ops=[], args=ham_args)

# ======================
# Population Analysis
# ======================
# Calculate populations for all basis states
populations = {label: [] for label in basis_labels}
for state in result.states:
    for label, basis_state in zip(basis_labels, basis_states):
        # Use overlap() method to get complex amplitude
        amplitude = basis_state.overlap(state)
        pop = abs(amplitude)**2
        populations[label].append(pop)
# Get final state information
final_state = result.states[-1]
final_populations = {label: populations[label][-1] for label in basis_labels}

# ======================
# Visualization
# ======================
plt.figure(figsize=(12, 6))

# Plot basis state populations
plt.subplot(121)
for label in basis_labels:
    plt.plot(tlist, populations[label], label=label)
    
plt.title("Basis State Populations")
plt.xlabel("Time (µs)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)

# Plot density matrix of final state
plt.subplot(122)
rho_final = final_state * final_state.dag()
plt.imshow(np.real(rho_final.full()), cmap='viridis')
plt.colorbar(label="Population")
plt.title("Final State Density Matrix")
plt.xticks([0,1,2,3], basis_labels)
plt.yticks([0,1,2,3], basis_labels)

plt.tight_layout()
plt.show()

# Print final state information
print("\nFinal State Populations:")
for label in basis_labels:
    print(f"|{label}⟩: {final_populations[label]:.4f}")

print("\nFinal State Vector:")
print(final_state)