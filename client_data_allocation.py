import argparse
import os
import json
import random
import pandas as pd
import numpy as np

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Partition dataset by category into training and test sets.")
    parser.add_argument('--input_file', type=str, default="new-databricks-dolly-15k.json",
                        help="Path to the input JSON file (records oriented).")
    parser.add_argument('--output_dir', type=str, default="data",
                        help="Output directory to save partitioned datasets.")
    parser.add_argument('--test_samples', type=int, default=10,
                        help="Number of samples per category to use for the test set.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Read the dataset from JSON file
    df = pd.read_json(args.input_file, orient='records')
    
    # Sort by category for consistency and group the data by category
    df_sorted = df.sort_values(by=['category'])
    grouped = df_sorted.groupby('category')
    
    # For each category, sample a fixed number of rows for the test set.
    # The remaining rows will form the training set.
    test_dfs = []
    training_dfs = []
    for cat, group in grouped:
        group = group.reset_index(drop=True)
        # If there are fewer rows than test_samples, take all as test
        if len(group) < args.test_samples:
            test_sample = group.copy()
            train_sample = pd.DataFrame(columns=group.columns)
        else:
            test_sample = group.sample(n=args.test_samples, random_state=args.seed)
            train_sample = group.drop(test_sample.index)
        test_dfs.append(test_sample)
        training_dfs.append(train_sample)
    
    # Concatenate all category test and training samples
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    training_df = pd.concat(training_dfs).reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save global training and test datasets to JSON files
    training_records = training_df.to_dict(orient='records')
    test_records = test_df.to_dict(orient='records')
    with open(os.path.join(args.output_dir, "global_training.json"), 'w') as outfile:
        json.dump(training_records, outfile)
    with open(os.path.join(args.output_dir, "global_test.json"), 'w') as outfile:
        json.dump(test_records, outfile)
    
    # Save local training datasets for each category
    for cat, group in training_df.groupby('category'):
        cat_records = group.to_dict(orient='records')
        # Create a valid filename by replacing spaces if needed
        filename = f"local_training_{str(cat).replace(' ', '_')}.json"
        with open(os.path.join(args.output_dir, filename), 'w') as outfile:
            json.dump(cat_records, outfile)
    
    # Print the sample counts per category for training and test sets
    print("Sample counts per category:")
    print("\nTraining Set:")
    for cat, group in training_df.groupby('category'):
        print(f"Category '{cat}': {len(group)} samples")
    
    print("\nTest Set:")
    for cat, group in test_df.groupby('category'):
        print(f"Category '{cat}': {len(group)} samples")

if __name__ == "__main__":
    main()
