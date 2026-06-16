import pandas as pd
import numpy as np

# Read the CSV file
csv_path = "segmentation_results/dice_score_summary.csv"  # Update with your path
df = pd.read_csv(csv_path)

# Filter only successful entries
success_df = df[df['status'] == 'Success'].copy()
success_df['dice_score'] = pd.to_numeric(success_df['dice_score'])

# Find highest, lowest, and median
highest = success_df.loc[success_df['dice_score'].idxmax()]
lowest = success_df.loc[success_df['dice_score'].idxmin()]
median_score_value = success_df['dice_score'].median()
median_idx = (success_df['dice_score'] - median_score_value).abs().argsort()[:1]
median = success_df.loc[median_idx].iloc[0]

# Print results
print("\n" + "="*60)
print("HIGHEST DICE SCORE")
print("="*60)
print(f"PMID: {highest['pmid']}")
print(f"Dice Score: {highest['dice_score']:.6f}")

print("\n" + "="*60)
print("MEDIAN DICE SCORE")
print("="*60)
print(f"PMID: {median['pmid']}")
print(f"Dice Score: {median['dice_score']:.6f}")
print(f"(Median value of all scores: {median_score_value:.6f})")

print("\n" + "="*60)
print("LOWEST DICE SCORE")
print("="*60)
print(f"PMID: {lowest['pmid']}")
print(f"Dice Score: {lowest['dice_score']:.6f}")

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Mean: {success_df['dice_score'].mean():.6f}")
print(f"Std Dev: {success_df['dice_score'].std():.6f}")
print(f"Count: {len(success_df)}")
