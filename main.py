import numpy as np
from qiskit import QuantumCircuit, Aer, execute
import matplotlib.pyplot as plt


class Gross_Pitaevskii_Simulator:
    def __init__(self, N, L, h_bar=1.0, m=1.0, g=1.0, dt=0.01, num_steps=100):
        self.N = N
        self.L = L
        self.h_bar = h_bar
        self.m = m
        self.g = g
        self.dt = dt
        self.num_steps = num_steps

        # Discretize spatial grid
        self.x = np.linspace (0, L, N)
        self.dx = L / N

        # Initialize wave function
        self.psi = np.sqrt (2 / L) * np.sin (np.pi * self.x / L)
        # Normalize the wave function
        self.psi /= np.sqrt (np.sum (np.abs (self.psi) ** 2))

        # Initialize quantum circuit
        self.qc = QuantumCircuit (N, N)


def encode_wave_function(self):
    # Normalize the wave function
    self.psi /= np.sqrt (np.sum (np.abs (self.psi) ** 2))

    for i, amp in enumerate (self.psi):
        self.qc.initialize ([amp, 0], i)


def encode_kinetic_energy(self):
    for i in range (self.N - 1):
        self.qc.rzz (-self.dt * self.h_bar / (2 * self.m * self.dx ** 2), i, i + 1)
        self.qc.rzz (-self.dt * self.h_bar / (2 * self.m * self.dx ** 2), i + 1, i)


def encode_potential_energy(self, omega):
    for i, xi in enumerate (self.x):
        self.qc.rz (self.dt * 0.5 * self.m * omega ** 2 * xi ** 2 / self.h_bar, i)


def encode_interaction_term(self):
    self.qc.rzz (-self.dt * self.g, range (self.N), range (self.N))


def run_simulation(self, omega):
    backend = Aer.get_backend ('statevector_simulator')

    for step in range (self.num_steps):
        self.encode_wave_function ()
        self.encode_kinetic_energy ()
        self.encode_potential_energy (omega)
        self.encode_interaction_term ()

        result = execute (self.qc, backend).result ()
        statevector = result.get_statevector ()

        # Extract updated wave function from the state vector
        self.psi = statevector [:self.N]

        # Normalize the wave function
        self.psi /= np.sqrt (np.sum (np.abs (self.psi) ** 2))

        # Visualization (optional)
        if step % 10 == 0:
            plt.plot (self.x, np.abs (self.psi) ** 2)
            plt.title (f'Time Step: {step}')
            plt.xlabel ('Position')
            plt.ylabel ('Probability Density')
            plt.show ()


# Example usage
if __name__ == '__main__':
    # Initialize the simulator
    simulator = Gross_Pitaevskii_Simulator (N=10, L=1.0, h_bar=1.0, m=1.0, g=1.0, dt=0.01, num_steps=100)

    # Run the simulation
    simulator

    # Plot the final wave function
    plt.plot (simulator.x, np.abs (simulator.psi) ** 2),
    plt.title ('Final Wave Function')
    plt.xlabel ('Position')
    plt.ylabel ('Probability Density')
    plt.show ()
