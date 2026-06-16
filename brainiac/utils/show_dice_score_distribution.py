import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Read the CSV file
csv_path = "segmentation_results/dice_score_summary.csv"  # Update with your path
df = pd.read_csv(csv_path)

# Filter successful entries
success_df = df[df['status'] == 'Success'].copy()
success_df['dice_score'] = pd.to_numeric(success_df['dice_score'])

# Create smooth histogram
fig, ax = plt.subplots(figsize=(10, 6))

# Use automatic bin selection for optimal smoothness
n_bins = int(np.sqrt(len(success_df))) * 2  # More bins for smoother look

# Plot histogram
ax.hist(success_df['dice_score'], bins=n_bins, alpha=0.5,
        color='steelblue', edgecolor='white', linewidth=0.5, density=True)

# Add smooth KDE
kde = gaussian_kde(success_df['dice_score'])
x_range = np.linspace(0, 1, 200)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='Smoothed density', alpha=0.8)

# Add mean and median
mean_val = success_df['dice_score'].mean()
median_val = success_df['dice_score'].median()
ax.axvline(mean_val, color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(median_val, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.7)

# Clean labels
ax.set_xlabel('Dice Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Distribution of Dice Scores (n={len(success_df)})', fontsize=14, pad=15)
ax.set_xlim(0, 1)
ax.grid(True, alpha=0.15)

# Simple legend
ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nn: {len(success_df)}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Print key stats
print(f"\nDice Score Statistics (n={len(success_df)}):")
print(f"  Mean: {mean_val:.4f}")
print(f"  Median: {median_val:.4f}")
print(f"  Std: {success_df['dice_score'].std():.4f}")
print(f"  Range: [{success_df['dice_score'].min():.4f}, {success_df['dice_score'].max():.4f}]")
