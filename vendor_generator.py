import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

plot = False

# Ensure output folder
os.makedirs("data", exist_ok=True)


# Parameters
num_vendors = 40

# local_fraction = 0.5

# rho = 1.3     # multiplier for digital delivery delay (based on avg. square distance ~ 0.521)
# sigma = 0.05  # noise in digital delivery time

# # Economics
# cost_mean = 0.7
# cost_std = 0.1
# margin_mean = 0.2
# margin_std = 0.05

# def generate_vendors(num_vendors, local_fraction=0.5, rho=1.3, sigma=0.05, cost_mean=0.7, cost_std=0.1, margin_mean=0.2, margin_std=0.05):
#     vendors = []
#     for i in range(num_vendors):
#         is_local = i < int(num_vendors * local_fraction)
#         vendor_type = 'local' if is_local else 'digital'
    
#         # Basic economics
#         cost = np.clip(np.random.normal(cost_mean, cost_std), 0.3, 1.5)
#         margin = np.clip(np.random.normal(margin_mean, margin_std), 0.05, 0.5)
#         price = cost + margin
    
#         if vendor_type == 'local':
#             x = np.random.rand()
#             y = np.random.rand()
#         else:
#             x = None
#             y = None
        
#         # Delivery time logic
#         if vendor_type == 'local':
#             delivery_time = 0.0  # handled via spatial urgency in U_ij
#         else:
#             delivery_time = np.random.normal(loc=rho * 0.521, scale=sigma)
#             delivery_time = max(0.1, delivery_time)  # no negative delays
    
        
#         quality = np.random.normal(loc=0.0, scale=0.5)  # mean 0, adds quality noise
#         rating = 4.5 - 1.2 * price + quality + np.random.normal(0, 0.3)
    
#         vendors.append({
#             'id': i,
#             'type': vendor_type,
#             'location_x': x,
#             'location_y': y,
#             'cost': cost,
#             'price': price,
#             'urgency_delay': delivery_time,
#             'rating': rating
#         })
    
#     # Save to CSV
#     df = pd.DataFrame(vendors)
#     df.to_csv("data/vendors.csv", index=False)
#     print("✅ Saved vendors to data/vendors.csv")

def generate_vendors(
    num_vendors,
    local_fraction=0.5,
    rho=1.3,
    sigma=0.30,
    cost_mean=0.70,
    cost_std=0.10,
    margin_mean=0.20,
    margin_std=0.05,
    # rating controls (price-only by default)
    rating_intercept=4.0,
    rating_slope=-1.0,      # negative → higher price lowers rating
    rating_noise=0.40,
    quality_std=0.0,        # set >0 to reintroduce an independent “quality” term
    # value guards
    cost_clip=(0.10, 5.00),
    margin_clip=(0.00, 2.00),
    price_clip=(0.10, 7.50),
):
    """
    Generate vendors with:
      - locals: real (x,y) in [0,1]^2, urgency_delay=NaN (distance used later)
      - digitals: (x,y)=NaN, urgency_delay=τ_j ~ max(0.1, N(rho*0.521, sigma))
      - rating default: price-only with Gaussian noise; optional independent quality via quality_std
    Saves to data/vendors.csv and returns the DataFrame.
    """
    M_local = int(round(num_vendors * local_fraction))
    types = np.array(['local'] * M_local + ['digital'] * (num_vendors - M_local))

    # --- economics ---
    cost   = np.clip(np.random.normal(cost_mean,   cost_std,   size=num_vendors), *cost_clip)
    margin = np.clip(np.random.normal(margin_mean, margin_std, size=num_vendors), *margin_clip)
    price  = np.clip(cost + margin, *price_clip)

    # --- geometry & latency ---
    loc_x = np.empty(num_vendors); loc_y = np.empty(num_vendors)
    loc_x[:] = np.nan;             loc_y[:] = np.nan
    # locals get real positions
    if M_local > 0:
        loc_x[:M_local] = np.random.rand(M_local)
        loc_y[:M_local] = np.random.rand(M_local)

    # τ_j for digitals, NaN for locals
    urgency_delay = np.empty(num_vendors); urgency_delay[:] = np.nan
    if num_vendors > M_local:
        tau = np.random.normal(loc=rho * 0.521, scale=sigma, size=num_vendors - M_local)
        urgency_delay[M_local:] = np.maximum(0.1, tau)

    # --- rating (price-only baseline, optional quality) ---
    quality = np.random.normal(0.0, quality_std, size=num_vendors) if quality_std > 0 else 0.0
    rating  = rating_intercept + rating_slope * price + quality + np.random.normal(0.0, rating_noise, size=num_vendors)
    rating  = np.clip(rating, 1.0, 5.0)

    df = pd.DataFrame({
        "id": np.arange(num_vendors, dtype=int),
        "type": types,
        "is_local": (types == 'local').astype(int),
        "location_x": loc_x,
        "location_y": loc_y,
        "cost": cost,
        "price": price,
        "urgency_delay": urgency_delay,  # used only if type=='digital'
        "rating": rating,
    })

    # Persist
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/vendors.csv", index=False)
    return df


# generate_vendors(num_vendors)


if __name__ == "__main__":
    df = generate_vendors(num_vendors)
    if plot:
        # --- Visualization ---
        sns.set(style="whitegrid")
        
        # 1. Spatial distribution
        plt.figure(figsize=(6, 6))
        for vendor_type, color in [('local', 'green'), ('digital', 'blue')]:
            subset = df[df['type'] == vendor_type]
            plt.scatter(subset['location_x'], subset['location_y'], label=vendor_type, alpha=0.6, c=color, edgecolors='k')
        
        plt.title("Vendor Spatial Distribution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("square")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("data/vendor_locations.png")
        print("📍 Saved vendor location plot to data/vendor_locations.png")
        
        # 2. Distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        params = ['price', 'urgency_delay', 'rating', 'cost']
        titles = ['Price', 'Urgency Delay', 'Rating', 'Operational Cost']
        
        for ax, param, title in zip(axes.flat, params, titles):
            sns.histplot(df[param], bins=20, kde=True, ax=ax, color="orange")
            ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig("data/vendor_distributions.png")
        print("📊 Saved vendor distribution plot to data/vendor_distributions.png")
