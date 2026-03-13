import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import KDTree
import time
import argparse
import os
from extract_velocity import extract_latent_vectors

def load_vector_field(trajectory_file):
    """
    Loads the dense GP trajectories and extracts the Position (Z) and Velocity (V) matrices.
    """
    print(f"Loading dense vector field from {trajectory_file}...")
    start_time = time.time()
    
    # Read the full high-resolution dataset
    df_traj = pd.read_csv(trajectory_file)
    
    # Identify the latent dimension Z columns and Velocity V columns
    z_cols = [col for col in df_traj.columns if col.startswith('z_mean_')]
    v_cols = [col for col in df_traj.columns if col.startswith('v_')]
    
    assert len(z_cols) == len(v_cols), "Mismatch in number of latent dimensions and velocity dimensions."
    
    Z = df_traj[z_cols].values
    V = df_traj[v_cols].values
    
    print(f"Loaded {Z.shape[0]} continuous states. Shape Z: {Z.shape}, Shape V: {V.shape}")
    print(f"Loading took {time.time() - start_time:.2f} seconds.")
    
    return Z, V

def build_kd_tree(Z):
    """
    Builds a spatial KD-Tree for O(log N) nearest neighbor quering in the 8D latent space.
    """
    print("Partitioning the state space into a KD-Tree...")
    start_time = time.time()
    
    tree = KDTree(Z, leaf_size=40)
    
    print(f"KD-Tree constructed in {time.time() - start_time:.2f} seconds.")
    return tree

def infer_velocity(z_new, tree, V_hist, k=500, sigma=1.0):
    """
    Queries the KD-Tree for the k nearest historical states to z_new, 
    and computes the distance-weighted average of their analytical velocities.
    """
    # 1. Query the KD-Tree
    # z_new must be 2D array: (1, 8)
    z_new_2d = z_new.reshape(1, -1)
    
    dist, ind = tree.query(z_new_2d, k=k)
    
    # Extract distances and indices (flatten from (1, k) to (k,))
    distances = dist[0]
    indices = ind[0]
    
    # 2. Extract historical velocities of neighbors
    V_neighbors = V_hist[indices]  # Shape: (k, 8)
    
    # 3. Compute Gaussian RBF Weights
    # w_i = exp(-d_i^2 / (2 * sigma^2))
    weights = np.exp(-(distances ** 2) / (2 * (sigma ** 2)))
    
    # 4. Compute weighted average velocity
    # Shape: (8,)
    v_inferred = np.average(V_neighbors, axis=0, weights=weights)
    
    return v_inferred, distances, weights

def simulate_clinical_visit(model_path, data_path, trajectory_path):
    """
    Simulates a new patient completing a questionnaire, encoding into the VAE,
    and inferring their instantaneous biological decay rate from the KD-Tree.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("--- Phase 9: Empirical Vector Field Inference ---")
    
    # 1. Build the Global State Space Index
    Z, V = load_vector_field(trajectory_path)
    tree = build_kd_tree(Z)
    
    # 2. Simulate "Patient X" encoding a cross-sectional questionnaire
    print("\nSimulating 'Patient X' Cross-Sectional VAE Encoding...")
    df_latent, z_cols = extract_latent_vectors(model_path, data_path, device)
    
    # We pick a random row from our *sparse* encodings to act as out-of-sample Patient X
    np.random.seed(42)
    random_idx = np.random.randint(0, len(df_latent))
    patient_x_data = df_latent.iloc[random_idx]
    
    z_new = patient_x_data[z_cols].values.astype(np.float64)
    patient_id = (patient_x_data['cunicah'], patient_x_data['np'])
    age_t = patient_x_data['t']
    
    print(f"Patient X Identifiers (cunicah, np): {patient_id}, Time t: {age_t:.1f}")
    print(f"Encoded Latent Location (z_new): {np.round(z_new, 3)}")
    
    # 3. Query the Empirical Vector Field
    print("\nInferring Biological Velocity from the State Space...")
    
    # Hyperparameters from protocol
    K_NEIGHBORS = 500
    SIGMA_KERNEL = 0.5  # Controls width of the "blur" / local neighborhood reach
    
    start_time = time.time()
    v_inferred, _, _ = infer_velocity(z_new, tree, V, k=K_NEIGHBORS, sigma=SIGMA_KERNEL)
    inference_time = time.time() - start_time
    
    print(f"Inferred Real-Time Velocity (v_new): {np.round(v_inferred, 4)}")
    print(f"Query completed in {inference_time * 1000:.2f} milliseconds.")
    
    # 4. Clinical Translation
    print("\n--- Clinical Translation ---")
    
    # Compute the systemic aging magnitude
    aging_magnitude = np.linalg.norm(v_inferred)
    print(f"Systemic Biological Aging Rate ||v_new|| : {aging_magnitude:.4f}")
    
    # Simulated Risk Stratification
    # (In a real system, these thresholds would be predefined by your Phase 5 Cox model Quartiles)
    print("\nRisk Stratification (Simulated):")
    if aging_magnitude > 0.8:
        print("-> STATUS: Q4 Fast Ager (High Risk). Projected trajectory dictates early intervention.")
    elif aging_magnitude < 0.3:
        print("-> STATUS: Q1 Slow Ager (Low Risk). Favorable resilience phenotype.")
    else:
        print("-> STATUS: Q2/Q3 Average Ager. Standard longitudinal monitoring recommended.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer biological velocity from a continuous empirical state space.")
    parser.add_argument('--model', default='beta_vae_model.pth', help='Path to VAE model')
    parser.add_argument('--data', default='frailty_index_data.csv', help='Path to raw dataset')
    parser.add_argument('--traj', default='latent_velocity_trajectory.csv', help='Path to continuous GP fields')
    
    args = parser.parse_args()
    
    # Adjust paths if run from root directory
    if not os.path.exists(args.model) and os.path.exists(os.path.join('latent_velocity', args.model)):
        args.model = os.path.join('latent_velocity', args.model)
    if not os.path.exists(args.data) and os.path.exists(os.path.join('latent_velocity', args.data)):
        args.data = os.path.join('latent_velocity', args.data)
    if not os.path.exists(args.traj) and os.path.exists(os.path.join('latent_velocity', args.traj)):
        args.traj = os.path.join('latent_velocity', args.traj)
    
    simulate_clinical_visit(args.model, args.data, args.traj)
