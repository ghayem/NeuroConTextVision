import pandas as pd
import numpy as np
import argparse

def get_top_n_dice_scores(csv_path, top_n=5):
    """Get top N PMIDs with highest Dice scores."""
    df = pd.read_csv(csv_path)
    success_df = df[df['status'] == 'Success'].copy()
    success_df['dice_score'] = pd.to_numeric(success_df['dice_score'])
    top_n_df = success_df.nlargest(top_n, 'dice_score')[['pmid', 'dice_score']].reset_index(drop=True)
    
    print("\n" + "="*60)
    print(f"TOP {top_n} PMIDS WITH HIGHEST DICE SCORES")
    print("="*60)
    for idx, row in top_n_df.iterrows():
        print(f"\n{idx+1}. PMID: {row['pmid']}")
        print(f"   Dice Score: {row['dice_score']:.6f}")
    
    return top_n_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get top N Dice scores from segmentation results')
    parser.add_argument('--csv', type=str, default='segmentation_results/dice_score_summary.csv',
                        help='Path to CSV file')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of top entries to display')
    
    args = parser.parse_args()
    result = get_top_n_dice_scores(args.csv, args.top_n)
    
    # Output as simple list for scripting
    print("\n" + "="*60)
    print("SIMPLE OUTPUT (PMID, Dice_Score)")
    print("="*60)
    for _, row in result.iterrows():
        print(f"{row['pmid']},{row['dice_score']:.6f}")