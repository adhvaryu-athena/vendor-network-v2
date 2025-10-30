import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plot = True

# Set seed for reproducibility
np.random.seed(25)

# Parameters
num_customers = 100

# Hyperparameters (tune as needed; keep names stable)
ALPHA_MEAN, ALPHA_STD = 1.0, 0.30       # price sensitivity α
URGENCY_MEAN, URGENCY_STD = 1.0, 0.30   # delay sensitivity u
DELTA_MEAN, DELTA_STD = 1.0, 0.30       # habit weight δ
SOCIAL_MEAN, SOCIAL_STD = 0.30, 0.10    # noise scale s

# Crucial: correlation between α and u (negative → realistic time–money tradeoff)
ALPHA_URGENCY_RHO = -0.5


# Ensure data folder exists
os.makedirs("data", exist_ok=True)

def generate_customers(num_customers):
    # ---- Locations in [0,1]^2 ----
    loc_x = np.clip(np.random.normal(loc=0.5, scale=0.2, size=num_customers), 0.0, 1.0)
    loc_y = np.clip(np.random.normal(loc=0.5, scale=0.2, size=num_customers), 0.0, 1.0)

    # ---- Correlated (alpha, urgency) ----
    mu  = np.array([ALPHA_MEAN, URGENCY_MEAN])
    sd  = np.array([ALPHA_STD, URGENCY_STD])
    cov = np.array([[sd[0]**2,              ALPHA_URGENCY_RHO*sd[0]*sd[1]],
                    [ALPHA_URGENCY_RHO*sd[0]*sd[1], sd[1]**2]])
    a_u = np.random.multivariate_normal(mu, cov, size=num_customers)
    alpha   = np.clip(a_u[:, 0], 0.05, 3.0)
    urgency = np.clip(a_u[:, 1], 0.05, 3.0)

    # ---- Habit (δ) and noise scale (s) ----
    delta  = np.clip(np.random.normal(DELTA_MEAN,  DELTA_STD,  size=num_customers), 0.05, 3.0)
    social = np.clip(np.random.normal(SOCIAL_MEAN, SOCIAL_STD, size=num_customers), 0.00, 2.0)

    # ---- Assemble & save ----
    df = pd.DataFrame({
        "id": np.arange(num_customers, dtype=int),
        "location_x": loc_x,
        "location_y": loc_y,
        "alpha": alpha,
        "urgency": urgency,
        "delta": delta,
        "social_influence": social,
    })
    df.to_csv("data/customers.csv", index=False)
    return df

if __name__ == "__main__":
    df = generate_customers(num_customers)
    if plot:
        # --- Visualization ---
        sns.set(style="whitegrid")
        
        # 1. Spatial scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(df['location_x'], df['location_y'], alpha=0.6, c="purple", edgecolors='k')
        plt.title("Customer Spatial Distribution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("square")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("data/customer_locations.png")
        print("📍 Saved customer location plot to data/customer_locations.png")
        
        # 2. Histograms
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        params = ['alpha', 'urgency', 'delta', 'social_influence']
        titles = ['Price Sensitivity (α)', 'Urgency Sensitivity (u)', 'Loyalty (δ)', 'Social Influence (s)']
        
        for ax, param, title in zip(axes.flat, params, titles):
            sns.histplot(df[param], bins=20, kde=True, ax=ax, color="skyblue")
            ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig("data/customer_distributions.png")
        print("📊 Saved customer distribution plot to data/customer_distributions.png")
