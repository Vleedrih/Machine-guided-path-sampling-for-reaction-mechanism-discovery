# Machine-Guided Path Sampling for Reaction Mechanism Discovery

## Project Overview

This repository contains the scientific software described in the Bachelor's thesis *"Machine-guided path sampling for reaction mechanism discovery"* by Vladyslav Palko. The software implements a machine learning-guided approach to Transition Path Sampling (TPS), aiming to efficiently discover and sample reaction mechanisms between metastable states. By training a neural network on an ensemble of reactive trajectories, the software learns an approximate committor function, which is then used to bias the selection of shooting points toward the most informative regions of path space.

---

## Scientific Background

The software relies on several advanced concepts in computational statistical physics:

* **Monte Carlo Phase-Space Sampling:** Utilizes the Metropolis algorithm to approximate target probability distributions, such as the Boltzmann distribution, by generating random samples.


* **Molecular Dynamics (MD) Sampling:** Evolves the system deterministically using Newton's equations of motion coupled with a stochastic thermostat (Langevin dynamics), specifically employing the BAOAB integration algorithm.


* **Transition Path Ensemble (TPE):** Instead of focusing on stable basins, the software samples the constrained ensemble of reactive trajectories connecting two metastable states, $A$ and $B$.


* **Shooting Algorithm:** A Monte Carlo procedure in path space that generates new candidate trajectories by applying local momentum perturbations to an existing reactive path and integrating forward and backward in time.


* **Machine-Guided Committor Construction:** A feed-forward neural network approximates the committor function, $P_B(q)$, predicting the probability that a trajectory starting from configuration $q$ will reach state $B$ before state $A$. This learned score steers the shooting point selection to the dynamical bottleneck.



---

## Features

* **Custom Simulation Engine:** Includes modular classes for basic Monte Carlo and Molecular Dynamics simulations for predefined 2D potentials (e.g., Double-well, Wolfe-Quapp).


* **Transition Path Extraction:** Automatically searches for and extracts valid reactive paths connecting user-defined stable regions.


* **Shooting Algorithm Implementation:** Modifies paths via momentum perturbations to sample the TPE.


* **Neural Network Committor:** Integrates PyTorch to define and train a custom neural network (`CommittorNet`) that learns transition likelihoods from trajectory data.


* **Biomolecular Simulation Support:** Hooks into the OpenMM library to simulate real molecules like alanine dipeptide in explicit water solvents.


* **Hyperparameter Optimization:** Utilizes Optuna to systematically explore and identify optimal model architectures, learning rates, and batch sizes.



---

## Software Architecture

The software is highly object-oriented, structured around several core classes:

### Base Classes and Simulators

* `system`: The base class containing physical and simulation parameters such as `num_particles`, `num_steps`, `dimension`, `temperature`, and the potential energy scaling factor `alpha`. It includes methods like `what_is_potential_energy` and `what_is_force`.


* `MonteCarloSimulator`: A subclass of `system` that adds `beta` ($\beta = \frac{1}{T}$) and `step_size`. It implements the Metropolis acceptance criterion within the `one_step` and `run` methods.


* `Molecular_dynamics`: A subclass of `system` utilizing the BAOAB algorithm. It introduces properties such as the time step `dt`, friction coefficient `gamma`, particle mass `m`, and thermostat variable `zeta`.



### Path Sampling and Machine Learning

* `Transition_Path_Ensemble`: A subclass inheriting from both `Molecular_dynamics` and `MonteCarloSimulator`. It includes a boolean flag `molec_dynam_step` to toggle the integration method. Key methods include `extract_original_transition_path`, `Create_a_transition_path_with_shooting`, and `create_committor`.


* `CommittorNet`: A subclass of `torch.nn.Module` that defines the neural network. It features a customizable number of hidden `torch.nn.Linear` layers with SiLU activation functions and a `train_step` method for optimizing weights using an Adam optimizer.



---

## Requirements and Dependencies

To execute the software successfully, the following Python libraries are strictly required:

* Numpy (version 2.3.5 or higher)


* Matplotlib (version 3.10.8 or higher)


* Numba (version 0.63.1 or higher)


* Torch (version 2.12.0 or higher)


* Optuna (version 4.8.0 or higher)


* Tqdm (version 4.67.1 or higher)


* h5py (version 3.15.1 or higher)


* MDTraj (version 1.11.1.post1 or higher)


* OpenMM (version 8.4.0 or higher)



---

## Installation

*(Inferred based on dependencies)*
Clone the repository and install the necessary dependencies via `pip` or `conda`. 

---

## Workflow Overview

1. **System Definition:** Initialize the physical system by defining the potential energy function (e.g., Double-well or Wolfe-Quapp) and stable state boundary criteria (e.g., specific ellipses in 2D space).


2. **Equilibrium Sampling:** Run standard MC or MD simulations for numerous steps (e.g., $10^6$ to $10^7$) to observe spontaneous transitions and extract initial reactive trajectories.


3. **Transition Path Sampling:** Utilize the `Create_a_transition_path_with_shooting` method to generate a statistically meaningful Transition Path Ensemble (TPE) from the initial path.


4. **Model Training:** Feed the sampled TPE into `CommittorNet`. The network learns to predict committor probabilities, recalculating selection probabilities iteratively.


5. **Evaluation:** Compare the model's predicted committor grid against brute-force estimated committor values to validate the network's understanding of the transition landscape.



---

## Output Files and Interpretation

* **Trajectories:** Saved particle positions and momenta at each step, managed via attributes like `trajectory` and `p_trajectories`.


* **Committor Grids:** 2D meshes (e.g., $x \in [-3, 3]$ and $y \in [-3, 3]$) mapping the predicted or calculated $P_B(q)$ value for specific configuration coordinates.


* **Model Metrics:** Evaluation functions plot the selection distribution, confusion plots, and training loss curves over defined epochs to interpret the neural network's accuracy.



---

## Performance Considerations and Limitations

* **Mass and Rejection Rates:** In 2D MD simulations, using a particle mass below 1 resulted in excessively high rejection rates (up to 80%) because the thermal momentum was insufficient to escape stable regions; setting the mass to 10 greatly improved the acceptance rate.


* **Algorithmic Bias:** In the biomolecule study, shooting point velocities were left unperturbed, which may have introduced a bias into the training data, distorting the committor predictions near the stable states.


* **Hyperparameter Sensitivity:** The software's learning efficiency is highly sensitive to the choice of hyperparameters such as the selection distribution regularization parameter ($\lambda$) and the efficiency threshold ($\alpha_0$).



                          
