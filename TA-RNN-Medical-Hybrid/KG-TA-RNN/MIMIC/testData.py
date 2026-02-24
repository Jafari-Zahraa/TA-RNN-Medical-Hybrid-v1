import pickle
import numpy as np

# ==============================
# Load generated PKL files
# ==============================

with open('DataScience/longitudinal_data.pkl', 'rb') as f:
    X = pickle.load(f)

with open('DataScience/label.pkl', 'rb') as f:
    y = pickle.load(f)

with open('DataScience/elapsed_data.pkl', 'rb') as f:
    elapsed = pickle.load(f)

with open('DataScience/demographic_data.pkl', 'rb') as f:
    demo = pickle.load(f)

# ==============================
# Print shapes
# ==============================

print("===== SHAPES =====")
print("X (longitudinal):", X.shape)        # (N, T, F)
print("y (mortality):  ", y.shape)         # (N, T, 1)
print("elapsed:        ", elapsed.shape)   # (N, T, 1)
print("demographic:    ", demo.shape)      # (N, D)

# ==============================
# Show one real patient
# ==============================

idx = 0   # first patient

print("\n===== SAMPLE PATIENT =====")

print("\nLongitudinal X [T x F]:")
print(X[idx])

print("\nMortality y [T x 1]:")
print(y[idx])

print("\nElapsed time [months]:")
print(elapsed[idx])

print("\nDemographic vector:")
print(demo[idx])

# ==============================
# Extra sanity checks
# ==============================

print("\n===== SANITY CHECKS =====")
print("Unique elapsed values:", np.unique(elapsed[idx][elapsed[idx] != -1]))
print("Padding value in X:", np.any(X == -1))
