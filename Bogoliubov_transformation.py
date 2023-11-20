from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit_aer.backends import qasm_simulator
import numpy as np
import matplotlib.pyplot as plt


class BECQuantumSimulator:
    def __init__(self, n_particles, interaction_strength):
        self.n_particles = n_particles
        self.interaction_strength = interaction_strength
        self.qc = QuantumCircuit (n_particles)

    def apply_bogoliubov_transformation(self):
        for i in range (self.n_particles):
            for j in range (i + 1, self.n_particles):
                self.qc.crz (2 * self.interaction_strength, i, j)
                self.qc.cx (i, j)
                self.qc.crz (-2 * self.interaction_strength, i, j)
                self.qc.cx (i, j)

    def apply_hamiltonian_terms(self):
        for i in range (self.n_particles):
            self.qc.rx (0.1, i)  # Kinetic energy term (adjust parameter)
            self.qc.rz (0.2, i)  # Harmonic trap potential term (adjust parameter)

    def run_simulation(self):
        self.qc.measure_all ()  # Add measurement step
        backend = Aer.get_backend ('qasm_simulator')
        job = execute (self.qc, backend)
        result = job.result ()
        return result.get_counts ()


class Visualizer:
    @staticmethod
    def plot_histogram(counts):
        plot_histogram (counts).show ()


# Example Usage:
n_particles = 3
interaction_strength = 0.1

bec_simulator = BECQuantumSimulator (n_particles, interaction_strength)
bec_simulator.apply_bogoliubov_transformation ()
bec_simulator.apply_hamiltonian_terms ()

counts = bec_simulator.run_simulation ()

visualizer = Visualizer ()
# visualizer.plot_histogram (counts)

# Visualize the quantum circuit
# bec_simulator.qc.draw ('mpl').show ()

# Calculate expectation values for specific observables
expectation_values = execute (bec_simulator.qc, qasm_simulator.QasmSimulator ()).result ().get_counts ()
print ("Expectation Values:", expectation_values)


# Make a function that calculates the energy of the system given the expectation values and other parameters
def calculate_energy(expectation_values, n_particles, interaction_strength):
    energy = 0
    for state, count in expectation_values.items ():
        energy += count * (state.count ('1') - n_particles / 2) ** 2
    energy *= interaction_strength
    return energy


print ("Energy:", calculate_energy (expectation_values, n_particles, interaction_strength))

# Calculate the energy for different values of the interaction strength
energies = []
interaction_strengths = np.linspace (0, 1, 10)
for interaction_strength in interaction_strengths:
    bec_simulator = BECQuantumSimulator (n_particles, interaction_strength)
    bec_simulator.apply_bogoliubov_transformation ()
    bec_simulator.apply_hamiltonian_terms ()
    bec_simulator.qc.measure_all()  # Add measurement step
    expectation_values = execute (bec_simulator.qc, qasm_simulator.QasmSimulator ()).result ().get_counts ()
    energy = calculate_energy(expectation_values, n_particles, interaction_strength)
    energies.append(energy)


# Plot the energy as a function of the interaction strength
plt.plot (interaction_strengths, energies)
plt.title ("Bogoliubov Energy")
plt.xlabel ("Interaction Strength")
plt.ylabel ("Energy")
plt.show ()
