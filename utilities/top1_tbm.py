import time

import pandas as pd
import numpy as np

import random
from Bio import pairwise2
from Bio.Seq import Seq

from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings('ignore')

print("\nLoading data files...")
train_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
valid_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
test_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
valid_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')

print(f"Loaded {len(train_seqs)} training sequences, {len(valid_seqs)} validation sequences, and {len(test_seqs)} test sequences")

# /usr/local/lib/python3.10/dist-packages/Bio/pairwise2.py:278: BiopythonDeprecationWarning: Bio.pairwise2 has been deprecated, and we intend to remove it in a future release of Biopython. As an alternative, please consider using Bio.Align.PairwiseAligner as a replacement, and contact the Biopython developers if you still need the Bio.pairwise2 module.
#   warnings.warn(
# Loading data files...
# Loaded 844 training sequences, 12 validation sequences, and 12 test sequences


# train_seqs_v2 = pd.read_csv('/kaggle/input/extended-rna/train_sequences_v2.csv')
# train_labels_v2 = pd.read_csv('/kaggle/input/extended-rna/train_labels_v2.csv')

train_seqs_v2 = pd.read_csv('/kaggle/input/rna-cif-to-csv/rna_sequences.csv')
train_labels_v2 = pd.read_csv('/kaggle/input/rna-cif-to-csv/rna_coordinates.csv')


train_seqs_v2.head()



# target_id	sequence
# 0	2D19_A	GCUGAAGUGCACACGGC
# 1	6OXI_QA	GUUGGAGAGUUUGAUCCUGGCUCAGGGUGAACGCUGGCGGCGUGCC...
# 2	6OXI_QV	CGCGGGGUGGAGCAGCCUGGUAGCUCGUCGGGCUCAUAACCCGAAG...
# 3	6OXI_QX	CAAGGAGGUAAAAAUGU
# 4	6OXI_RA	AGAUGGUAAGGGCCCACGGUGGAUGCCUCGGCACCCGAGCCGAUGA...


train_labels_v2.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10135546 entries, 0 to 10135545
# Data columns (total 6 columns):
#  #   Column   Dtype  
# ---  ------   -----  
#  0   ID       object 
#  1   resname  object 
#  2   resid    int64  
#  3   x_1      float64
#  4   y_1      float64
#  5   z_1      float64
# dtypes: float64(3), int64(1), object(2)
# memory usage: 464.0+ MB


# import pandas as pd
# import numpy as np

# # 1. Check if v2 is larger than original
# print("Dataset Size Comparison:")
# print(f"train_seqs: {len(train_seqs)} rows → train_seqs_v2: {len(train_seqs_v2)} rows")
# print(f"train_labels: {len(train_labels)} rows → train_labels_v2: {len(train_labels_v2)} rows")
# print()

# # 2. Verify all train_seqs records exist in train_seqs_v2
# print("Checking if original sequences exist in v2...")
# original_target_ids = set(train_seqs['target_id'])
# extended_target_ids = set(train_seqs_v2['target_id'])
# all_targets_included = original_target_ids.issubset(extended_target_ids)

# print(f"Original train_seqs has {len(original_target_ids)} unique target_ids")
# print(f"All original target_ids found in train_seqs_v2: {all_targets_included}")

# if not all_targets_included:
#     missing_ids = original_target_ids - extended_target_ids
#     print(f"Missing {len(missing_ids)} target_ids")
#     if len(missing_ids) <= 5:
#         print(f"Missing IDs: {list(missing_ids)}")
#     else:
#         print(f"First 5 missing IDs: {list(missing_ids)[:5]}")
# print()

# # 3. Check for consistency in a sample of target_ids that exist in both datasets
# if all_targets_included:
#     print("Checking consistency of data between original and v2 sequences...")
#     # Sample a few target_ids that exist in both datasets
#     sample_size = min(5, len(original_target_ids))
#     sample_ids = np.random.choice(list(original_target_ids), sample_size, replace=False)
    
#     for target_id in sample_ids:
#         original_row = train_seqs[train_seqs['target_id'] == target_id].iloc[0]
#         extended_row = train_seqs_v2[train_seqs_v2['target_id'] == target_id].iloc[0]
        
#         # Compare important columns
#         sequence_match = original_row['sequence'] == extended_row['sequence']
#         print(f"Target ID {target_id}: Sequences match: {sequence_match}")
# print()

# # 4. Check if all train_labels IDs exist in train_labels_v2
# print("Checking labels dataset...")
# # Since labels dataset is large, we'll check a sample of IDs
# sample_size = min(1000, len(train_labels))
# sample_indices = np.random.choice(len(train_labels), sample_size, replace=False)
# sample_rows = train_labels.iloc[sample_indices]

# # Create a composite key for comparison (ID + resid)
# sample_rows['composite_key'] = sample_rows['ID'] + '_' + sample_rows['resid'].astype(str)
# train_labels_v2['composite_key'] = train_labels_v2['ID'] + '_' + train_labels_v2['resid'].astype(str)

# sample_keys = set(sample_rows['composite_key'])
# extended_keys = set(train_labels_v2['composite_key'])

# keys_found = sample_keys.issubset(extended_keys)
# if keys_found:
#     found_percentage = 100
# else:
#     intersection = sample_keys.intersection(extended_keys)
#     found_percentage = (len(intersection) / len(sample_keys)) * 100

# print(f"Sampled {len(sample_keys)} keys from train_labels")
# print(f"All sampled keys found in train_labels_v2: {keys_found} ({found_percentage:.2f}%)")

# if not keys_found:
#     missing_keys = sample_keys - extended_keys
#     print(f"Missing {len(missing_keys)} keys out of {len(sample_keys)} sampled")
#     if len(missing_keys) <= 5:
#         print(f"Missing keys: {list(missing_keys)}")
#     else:
#         print(f"First 5 missing keys: {list(missing_keys)[:5]}")
# print()

# # 5. Data type consistency check
# print("Data type consistency check:")
# print("train_seqs vs train_seqs_v2:")
# for col in train_seqs.columns:
#     print(f"  Column '{col}': {train_seqs[col].dtype} → {train_seqs_v2[col].dtype} - Match: {train_seqs[col].dtype == train_seqs_v2[col].dtype}")

# print("\ntrain_labels vs train_labels_v2:")
# for col in train_labels.columns:
#     if col != 'composite_key':  # Skip the key we created
#         print(f"  Column '{col}': {train_labels[col].dtype} → {train_labels_v2[col].dtype} - Match: {train_labels[col].dtype == train_labels_v2[col].dtype}")
# print()

# # 6. Check for missing values pattern
# print("Missing values comparison:")
# print("train_seqs vs train_seqs_v2:")
# for col in train_seqs.columns:
#     original_missing = train_seqs[col].isnull().sum() / len(train_seqs) * 100
#     extended_missing = train_seqs_v2[col].isnull().sum() / len(train_seqs_v2) * 100
#     print(f"  Column '{col}': {original_missing:.2f}% → {extended_missing:.2f}%")

# print("\ntrain_labels vs train_labels_v2:")
# for col in train_labels.columns:
#     if col != 'composite_key':  # Skip the key we created
#         original_missing = train_labels[col].isnull().sum() / len(train_labels) * 100
#         extended_missing = train_labels_v2[col].isnull().sum() / len(train_labels_v2) * 100
#         print(f"  Column '{col}': {original_missing:.2f}% → {extended_missing:.2f}%")
# print()

# # Clean up the temporary column we added
# if 'composite_key' in train_labels_v2.columns:
#     train_labels_v2.drop('composite_key', axis=1, inplace=True)

# # Final assessment
# print("FINAL ASSESSMENT:")
# print("-" * 50)
# seqs_extended_properly = all_targets_included and len(train_seqs_v2) > len(train_seqs)
# labels_extended_properly = found_percentage > 99 and len(train_labels_v2) > len(train_labels)

# if seqs_extended_properly and labels_extended_properly:
#     print("✓ PASS: Both train_seqs_v2 and train_labels_v2 appear to be proper extensions of the original datasets.")
#     print("✓ It should be safe to swap them.")
# else:
#     print("✗ ISSUES DETECTED:")
#     if not seqs_extended_properly:
#         print("  - train_seqs_v2 may not fully contain train_seqs data")
#     if not labels_extended_properly:
#         print("  - train_labels_v2 may not fully contain train_labels data")
#     print("✗ Recommend investigating the issues above before swapping datasets.")
# print("-" * 50)


import pandas as pd
import numpy as np

# Function to extend the original dataset with new records from v2
def extend_dataset(original_df, v2_df, key_columns, dataset_name):
    print(f"Extending {dataset_name}...")
    print(f"  Original size: {len(original_df)} rows")
    print(f"  v2 size: {len(v2_df)} rows")
    
    # Create a composite key for identification if multiple key columns
    if isinstance(key_columns, list) and len(key_columns) > 1:
        original_df['temp_key'] = original_df[key_columns].astype(str).agg('_'.join, axis=1)
        v2_df['temp_key'] = v2_df[key_columns].astype(str).agg('_'.join, axis=1)
        key_for_identification = 'temp_key'
    else:
        key_for_identification = key_columns[0] if isinstance(key_columns, list) else key_columns
    
    # Identify unique records in each dataset
    original_keys = set(original_df[key_for_identification])
    v2_keys = set(v2_df[key_for_identification])
    
    # Calculate stats
    keys_only_in_original = original_keys - v2_keys
    keys_only_in_v2 = v2_keys - original_keys 
    common_keys = original_keys.intersection(v2_keys)
    
    print(f"  Keys only in original: {len(keys_only_in_original)}")
    print(f"  Keys only in v2: {len(keys_only_in_v2)}")
    print(f"  Common keys: {len(common_keys)}")
    
    # Create a mask to filter v2 records that don't exist in original
    new_records_mask = ~v2_df[key_for_identification].isin(original_keys)
    new_records = v2_df[new_records_mask].copy()
    
    # Drop temporary key if it was created
    if key_for_identification == 'temp_key':
        new_records.drop('temp_key', axis=1, inplace=True)
        original_df.drop('temp_key', axis=1, inplace=True)
    
    # Combine original with new records from v2
    extended_df = pd.concat([original_df, new_records], ignore_index=True)
    
    # Report final sizes
    print(f"  New records added: {len(new_records)}")
    print(f"  Extended dataset size: {len(extended_df)} rows")
    print(f"  Verification - All original keys in extended dataset: {set(original_df[key_columns[0] if isinstance(key_columns, list) else key_columns]).issubset(set(extended_df[key_columns[0] if isinstance(key_columns, list) else key_columns]))}")
    
    # Check for missing values in key columns
    for col in extended_df.columns:
        original_missing = original_df[col].isnull().sum()
        extended_missing = extended_df[col].isnull().sum()
        if original_missing > 0 or extended_missing > 0:
            print(f"  Column '{col}': Missing values - Original: {original_missing}, Extended: {extended_missing}")
    
    # Clean up
    if key_for_identification == 'temp_key' and 'temp_key' in v2_df.columns:
        v2_df.drop('temp_key', axis=1, inplace=True)
        
    return extended_df

# 1. Extend train_seqs with train_seqs_v2
print("\n" + "="*50)
print("EXTENDING SEQUENCE DATASETS")
print("="*50)
train_seqs_extended = extend_dataset(
    train_seqs, 
    train_seqs_v2,
    ['target_id'],  # Using target_id as the unique identifier
    "train_seqs"
)

# 2. Extend train_labels with train_labels_v2
print("\n" + "="*50)
print("EXTENDING LABELS DATASETS")
print("="*50)
# For labels, we need a composite key of ID and resid
train_labels_extended = extend_dataset(
    train_labels,
    train_labels_v2,
    ['ID', 'resid'],  # Using composite key
    "train_labels"
)

# Verify relationships between extended datasets
print("\n" + "="*50)
print("VERIFYING RELATIONSHIPS")
print("="*50)

# Check if all sequence IDs have corresponding labels
seq_ids = set(train_seqs_extended['target_id'].unique())
label_ids = set(train_labels_extended['ID'].unique())

seq_ids_with_labels = seq_ids.intersection(label_ids)
seq_ids_without_labels = seq_ids - label_ids

print(f"Total unique sequence IDs: {len(seq_ids)}")
print(f"Sequence IDs with corresponding labels: {len(seq_ids_with_labels)} ({len(seq_ids_with_labels)/len(seq_ids)*100:.2f}%)")
print(f"Sequence IDs without corresponding labels: {len(seq_ids_without_labels)} ({len(seq_ids_without_labels)/len(seq_ids)*100:.2f}%)")

if len(seq_ids_without_labels) > 0:
    print("Sample of sequence IDs without labels (up to 5):")
    print(list(seq_ids_without_labels)[:5])

# Print summary of extended datasets
print("\n" + "="*50)
print("SUMMARY OF EXTENDED DATASETS")
print("="*50)
print(f"Original train_seqs: {len(train_seqs)} rows")
print(f"Original train_labels: {len(train_labels)} rows")
print(f"Extended train_seqs: {len(train_seqs_extended)} rows (+{len(train_seqs_extended)-len(train_seqs)})")
print(f"Extended train_labels: {len(train_labels_extended)} rows (+{len(train_labels_extended)-len(train_labels)})")

# Save the extended datasets (uncomment to save)
# train_seqs_extended.to_csv('train_seqs_combined.csv', index=False)
# train_labels_extended.to_csv('train_labels_combined.csv', index=False)

print("\n" + "="*50)
print("DONE! Extended datasets created.")
print("To save the datasets, uncomment the last two lines.")
print("="*50)


# ==================================================
# EXTENDING SEQUENCE DATASETS
# ==================================================
# Extending train_seqs...
#   Original size: 844 rows
#   v2 size: 18881 rows
#   Keys only in original: 65
#   Keys only in v2: 18102
#   Common keys: 779
#   New records added: 18102
#   Extended dataset size: 18946 rows
#   Verification - All original keys in extended dataset: True
#   Column 'temporal_cutoff': Missing values - Original: 0, Extended: 18102
#   Column 'description': Missing values - Original: 0, Extended: 18102
#   Column 'all_sequences': Missing values - Original: 5, Extended: 18107

# ==================================================
# EXTENDING LABELS DATASETS
# ==================================================
# Extending train_labels...
#   Original size: 137095 rows
#   v2 size: 10135546 rows
#   Keys only in original: 11896
#   Keys only in v2: 10010347
#   Common keys: 125199
#   New records added: 10010347
#   Extended dataset size: 10147442 rows
#   Verification - All original keys in extended dataset: True
#   Column 'x_1': Missing values - Original: 6145, Extended: 6145
#   Column 'y_1': Missing values - Original: 6145, Extended: 6145
#   Column 'z_1': Missing values - Original: 6145, Extended: 6145

# ==================================================
# VERIFYING RELATIONSHIPS
# ==================================================
# Total unique sequence IDs: 18946
# Sequence IDs with corresponding labels: 0 (0.00%)
# Sequence IDs without corresponding labels: 18946 (100.00%)
# Sample of sequence IDs without labels (up to 5):
# ['7WTW_2', '6M7K_B', '8T77_B', '5UYQ_V', '4U35_A']

# ==================================================
# SUMMARY OF EXTENDED DATASETS
# ==================================================
# Original train_seqs: 844 rows
# Original train_labels: 137095 rows
# Extended train_seqs: 18946 rows (+18102)
# Extended train_labels: 10147442 rows (+10010347)

# ==================================================
# DONE! Extended datasets created.
# To save the datasets, uncomment the last two lines.
# ==================================================


train_seqs_extended.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 18946 entries, 0 to 18945
# Data columns (total 5 columns):
#  #   Column           Non-Null Count  Dtype 
# ---  ------           --------------  ----- 
#  0   target_id        18946 non-null  object
#  1   sequence         18946 non-null  object
#  2   temporal_cutoff  844 non-null    object
#  3   description      844 non-null    object
#  4   all_sequences    839 non-null    object
# dtypes: object(5)
# memory usage: 740.2+ KB

train_labels_extended.info()


# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10147442 entries, 0 to 10147441
# Data columns (total 6 columns):
#  #   Column   Dtype  
# ---  ------   -----  
#  0   ID       object 
#  1   resname  object 
#  2   resid    int64  
#  3   x_1      float64
#  4   y_1      float64
#  5   z_1      float64
# dtypes: float64(3), int64(1), object(2)
# memory usage: 464.5+ MB


# # Get the first 500 sequences
# train_seqs_small = train_seqs_extended.iloc[:500].copy()

# # Extract base IDs from train_labels_extended once
# base_ids = train_labels_extended['ID'].str.rsplit('_', n=1).str[0]

# # Filter labels where the base ID is in our sequence IDs
# train_labels_small = train_labels_extended[base_ids.isin(train_seqs_small['target_id'])].copy()

# # Verify
# print(f"Number of sequences in train_seqs_small: {len(train_seqs_small)}")
# print(f"Total rows in train_labels_small: {len(train_labels_small)}")


def process_labels(labels_df):
    coords_dict = {}
    
    # Group by target ID and wrap with tqdm for progress tracking
    id_groups = labels_df.groupby(lambda x: labels_df['ID'][x].rsplit('_', 1)[0])
    for id_prefix, group in tqdm(id_groups, desc="Processing structures"):
        # Extract just the coordinates columns for the first structure (x_1, y_1, z_1)
        coords = []
        for _, row in group.sort_values('resid').iterrows():
            coords.append([row['x_1'], row['y_1'], row['z_1']])
        
        coords_dict[id_prefix] = np.array(coords)
    
    return coords_dict

train_coords_dict = process_labels(train_labels_extended)

# Processing structures: 100%|██████████| 18815/18815 [09:31<00:00, 32.92it/s]

from Bio.Seq import Seq
from Bio import pairwise2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=5):
    """
    Find similar RNA sequences using enhanced scoring and clustering for diversity.
    
    Improvements:
    - Multi-tier length filtering
    - Enhanced alignment scoring with multiple algorithms
    - RNA-specific structural features
    - Adaptive clustering
    """
    similar_seqs = []
    query_seq_obj = Seq(query_seq)
    query_features = _extract_enhanced_rna_features(query_seq)
    
    # Step 1: Enhanced candidate selection with multi-tier filtering
    for _, row in train_seqs_df.iterrows():
        target_id = row['target_id']
        train_seq = row['sequence']
        
        # Skip if coordinates not available
        if target_id not in train_coords_dict:
            continue
        
        # Multi-tier length filtering (more permissive for very short/long sequences)
        len_ratio = abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq))
        if len(query_seq) < 50 or len(train_seq) < 50:  # Short sequences - more permissive
            if len_ratio > 0.6:
                continue
        elif len(query_seq) > 1000 or len(train_seq) > 1000:  # Long sequences - stricter
            if len_ratio > 0.2:
                continue
        else:  # Medium sequences - original threshold
            if len_ratio > 0.4:
                continue
        
        # Calculate composite similarity score
        composite_score = _calculate_composite_similarity(query_seq, train_seq, query_features)
        
        if composite_score > 0:  # Only keep sequences with positive similarity
            similar_seqs.append((target_id, train_seq, composite_score, train_coords_dict[target_id]))
    
    # Sort by composite score and take top candidates
    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    
    # Adaptive candidate selection based on score distribution
    candidate_count = min(50, len(similar_seqs))  # Increased initial pool
    if len(similar_seqs) > 10:
        # Filter out sequences with very low scores (bottom 20%)
        score_threshold = np.percentile([x[2] for x in similar_seqs], 80)
        filtered_candidates = [x for x in similar_seqs if x[2] >= score_threshold]
        candidate_count = min(candidate_count, len(filtered_candidates))
        top_candidates = filtered_candidates[:candidate_count]
    else:
        top_candidates = similar_seqs[:candidate_count]
    
    # If we have fewer sequences than requested clusters, return all
    if len(top_candidates) <= top_n:
        return top_candidates[:top_n]
    
    # Step 2: Enhanced feature matrix for better clustering
    feature_matrix = []
    for _, seq, _, _ in top_candidates:
        features = _extract_enhanced_rna_features(seq)
        feature_matrix.append(features)
    
    feature_matrix = np.array(feature_matrix)
    
    # Step 3: Adaptive clustering
    n_clusters = min(top_n, len(top_candidates))
    
    # Use different clustering approach based on dataset size
    if len(top_candidates) >= 15:
        # K-means for larger datasets
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
    else:
        # Simple diversity-based selection for smaller datasets
        cluster_labels = _diversity_based_clustering(feature_matrix, n_clusters)
    
    # Step 4: Select best representative from each cluster
    final_results = []
    for cluster_id in range(n_clusters):
        cluster_sequences = [top_candidates[i] for i in range(len(top_candidates)) 
                           if cluster_labels[i] == cluster_id]
        
        if cluster_sequences:
            # Sort by composite score and take the best one
            cluster_sequences.sort(key=lambda x: x[2], reverse=True)
            final_results.append(cluster_sequences[0])
    
    # Sort final results by similarity score
    final_results.sort(key=lambda x: x[2], reverse=True)
    
    return final_results[:top_n]

def _calculate_composite_similarity(query_seq, train_seq, query_features):
    """
    Calculate composite similarity using multiple alignment methods and features.
    """
    query_seq_obj = Seq(query_seq)
    
    # 1. Global alignment (original method)
    global_alignments = pairwise2.align.globalms(query_seq_obj, train_seq, 2.9, -1, -10, -0.5, one_alignment_only=True)
    global_score = 0
    if global_alignments:
        alignment = global_alignments[0]
        global_score = alignment.score / (2 * min(len(query_seq), len(train_seq)))
    
    # 2. Local alignment for finding similar regions
    local_alignments = pairwise2.align.localms(query_seq_obj, train_seq, 2.9, -1, -10, -0.5, one_alignment_only=True)
    local_score = 0
    if local_alignments:
        alignment = local_alignments[0]
        local_score = alignment.score / (2 * min(len(query_seq), len(train_seq)))
    
    # 3. Feature-based similarity
    train_features = _extract_enhanced_rna_features(train_seq)
    feature_similarity = cosine_similarity([query_features], [train_features])[0][0]
    
    # 4. K-mer similarity for sequence motifs
    kmer_similarity = _calculate_kmer_similarity(query_seq, train_seq, k=3)
    
    # Weighted composite score
    composite_score = (
        0.4 * global_score + 
        0.3 * local_score + 
        0.2 * feature_similarity + 
        0.1 * kmer_similarity
    )
    
    return composite_score

def _calculate_kmer_similarity(seq1, seq2, k=3):
    """Calculate k-mer based similarity between sequences."""
    def get_kmers(seq, k):
        return set(seq[i:i+k] for i in range(len(seq) - k + 1))
    
    kmers1 = get_kmers(seq1.upper(), k)
    kmers2 = get_kmers(seq2.upper(), k)
    
    if not kmers1 or not kmers2:
        return 0
    
    intersection = len(kmers1.intersection(kmers2))
    union = len(kmers1.union(kmers2))
    
    return intersection / union if union > 0 else 0

def _diversity_based_clustering(feature_matrix, n_clusters):
    """Simple diversity-based clustering for small datasets."""
    n_samples = len(feature_matrix)
    cluster_labels = np.zeros(n_samples, dtype=int)
    
    if n_samples <= n_clusters:
        return np.arange(n_samples)
    
    # Select diverse representatives
    selected_indices = [0]  # Start with first sequence
    
    for cluster_id in range(1, n_clusters):
        max_min_distance = -1
        best_idx = -1
        
        for i in range(n_samples):
            if i in selected_indices:
                continue
            
            # Find minimum distance to already selected sequences
            min_distance = min(
                np.linalg.norm(feature_matrix[i] - feature_matrix[j]) 
                for j in selected_indices
            )
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_idx = i
        
        if best_idx != -1:
            selected_indices.append(best_idx)
    
    # Assign remaining sequences to closest cluster centers
    for i in range(n_samples):
        if i not in selected_indices:
            distances = [
                np.linalg.norm(feature_matrix[i] - feature_matrix[j]) 
                for j in selected_indices
            ]
            cluster_labels[i] = np.argmin(distances)
        else:
            cluster_labels[i] = selected_indices.index(i)
    
    return cluster_labels

def _extract_enhanced_rna_features(sequence):
    """
    Extract comprehensive RNA-specific features for better clustering and similarity.
    """
    seq = sequence.upper()
    features = []
    
    # 1. Basic nucleotide frequencies
    nucleotides = ['A', 'U', 'G', 'C']
    for nuc in nucleotides:
        freq = seq.count(nuc) / len(seq) if len(seq) > 0 else 0
        features.append(freq)
    
    # 2. Dinucleotide frequencies (reduced set - most important for RNA)
    important_dinucs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG', 'AA', 'UU', 'GG', 'CC']
    for dinuc in important_dinucs:
        count = 0
        for i in range(len(seq) - 1):
            if seq[i:i+2] == dinuc:
                count += 1
        freq = count / (len(seq) - 1) if len(seq) > 1 else 0
        features.append(freq)
    
    # 3. RNA secondary structure indicators
    gc_content = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
    au_content = (seq.count('A') + seq.count('U')) / len(seq) if len(seq) > 0 else 0
    purine_content = (seq.count('A') + seq.count('G')) / len(seq) if len(seq) > 0 else 0
    pyrimidine_content = (seq.count('U') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
    
    features.extend([gc_content, au_content, purine_content, pyrimidine_content])
    
    # 4. Sequence complexity measures
    length_normalized = min(len(seq) / 1000.0, 1.0)  # Capped normalization
    
    # Simple entropy calculation
    entropy = 0
    for nuc in nucleotides:
        freq = seq.count(nuc) / len(seq) if len(seq) > 0 else 0
        if freq > 0:
            entropy -= freq * np.log2(freq)
    entropy_normalized = entropy / 2.0  # Max entropy for 4 nucleotides is 2
    
    features.extend([length_normalized, entropy_normalized])
    
    # 5. Repetitive pattern detection
    repeat_content = _calculate_repeat_content(seq)
    features.append(repeat_content)
    
    return features

def _calculate_repeat_content(sequence):
    """Calculate the proportion of repetitive content in the sequence."""
    if len(sequence) < 6:
        return 0
    
    repeat_count = 0
    window_size = 3
    
    for i in range(len(sequence) - window_size + 1):
        motif = sequence[i:i + window_size]
        # Look for the same motif in the rest of the sequence
        for j in range(i + window_size, len(sequence) - window_size + 1):
            if sequence[j:j + window_size] == motif:
                repeat_count += 1
                break
    
    return repeat_count / (len(sequence) - window_size + 1) if len(sequence) > window_size else 0



def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    # Make a copy of coordinates to refine
    refined_coords = coordinates.copy()
    n_residues = len(sequence)
    
    # Calculate constraint strength (inverse of confidence)
    # High confidence templates receive gentler constraints
    constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))
    
    # 1. Sequential distance constraints (consecutive nucleotides)
    # More flexible distance range (statistical distribution from PDB)
    seq_min_dist = 5.5  # Minimum sequential distance
    seq_max_dist = 6.5  # Maximum sequential distance
    
    for i in range(n_residues - 1):
        current_pos = refined_coords[i]
        next_pos = refined_coords[i+1]
        
        # Calculate current distance
        current_dist = np.linalg.norm(next_pos - current_pos)
        
        # Only adjust if significantly outside expected range
        if current_dist < seq_min_dist or current_dist > seq_max_dist:
            # Calculate target distance (midpoint of range)
            target_dist = (seq_min_dist + seq_max_dist) / 2
            
            # Get direction vector
            direction = next_pos - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Apply partial adjustment based on constraint strength
            adjustment = (target_dist - current_dist) * constraint_strength
            
            # Only adjust the next position to preserve the overall fold
            refined_coords[i+1] = current_pos + direction * (current_dist + adjustment)
    
    # 2. Steric clash prevention (more conservative)
    min_allowed_distance = 3.8  # Minimum distance between non-consecutive C1' atoms
    
    # Calculate all pairwise distances
    dist_matrix = distance_matrix(refined_coords, refined_coords)
    
    # Find severe clashes (atoms too close)
    severe_clashes = np.where((dist_matrix < min_allowed_distance) & (dist_matrix > 0))
    
    # Fix severe clashes
    for idx in range(len(severe_clashes[0])):
        i, j = severe_clashes[0][idx], severe_clashes[1][idx]
        
        # Skip consecutive nucleotides and previously processed pairs
        if abs(i - j) <= 1 or i >= j:
            continue
            
        # Get current positions and distance
        pos_i = refined_coords[i]
        pos_j = refined_coords[j]
        current_dist = dist_matrix[i, j]
        
        # Calculate necessary adjustment but scale by constraint strength
        direction = pos_j - pos_i
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Calculate partial adjustment
        adjustment = (min_allowed_distance - current_dist) * constraint_strength
        
        # Move points apart
        refined_coords[i] = pos_i - direction * (adjustment / 2)
        refined_coords[j] = pos_j + direction * (adjustment / 2)
    
    # 3. Very light base-pair constraining (if confidence is low)
    if constraint_strength > 0.3:  # Only apply if template confidence is low
        # Simple Watson-Crick base pairs
        pairs = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        
        # Scan for potential base pairs
        for i in range(n_residues):
            base_i = sequence[i]
            complement = pairs.get(base_i)
            
            if not complement:
                continue
                
            # Look for complementary bases within a reasonable range
            for j in range(i + 3, min(i + 20, n_residues)):
                if sequence[j] == complement:
                    # Calculate current distance
                    current_dist = np.linalg.norm(refined_coords[i] - refined_coords[j])
                    
                    # Only consider if distance suggests potential pairing
                    if 8.0 < current_dist < 14.0:
                        # Target 10.5Å as generic base-pair C1'-C1' distance
                        target_dist = 10.5
                        
                        # Calculate very gentle adjustment (scaled by constraint_strength)
                        adjustment = (target_dist - current_dist) * (constraint_strength * 0.3)
                        
                        # Get direction vector
                        direction = refined_coords[j] - refined_coords[i]
                        direction = direction / (np.linalg.norm(direction) + 1e-10)
                        
                        # Apply very gentle adjustment to both positions
                        refined_coords[i] = refined_coords[i] - direction * (adjustment / 2)
                        refined_coords[j] = refined_coords[j] + direction * (adjustment / 2)
                        
                        # Only consider one potential pair per base (closest match)
                        break
    
    return refined_coords



def adapt_template_to_query(query_seq, template_seq, template_coords, alignment=None):
    if alignment is None:
        from Bio.Seq import Seq
        from Bio import pairwise2
        
        query_seq_obj = Seq(query_seq)
        template_seq_obj = Seq(template_seq)
        alignments = pairwise2.align.globalms(query_seq_obj, template_seq_obj, 2.9, -1, -10, -0.5, one_alignment_only=True)
        
        if not alignments:
            return generate_improved_rna_structure(query_seq)
            
        alignment = alignments[0]
    
    aligned_query = alignment.seqA
    aligned_template = alignment.seqB
    
    query_coords = np.zeros((len(query_seq), 3))
    query_coords.fill(np.nan)
    
    # Map template coordinates to query
    query_idx = 0
    template_idx = 0
    
    for i in range(len(aligned_query)):
        query_char = aligned_query[i]
        template_char = aligned_template[i]
        
        if query_char != '-' and template_char != '-':
            if template_idx < len(template_coords):
                query_coords[query_idx] = template_coords[template_idx]
            template_idx += 1
            query_idx += 1
        elif query_char != '-' and template_char == '-':
            query_idx += 1
        elif query_char == '-' and template_char != '-':
            template_idx += 1
    
    # IMPROVED GAP FILLING - maintains RNA backbone geometry
    backbone_distance = 5.9  # Typical C1'-C1' distance
    
    # Fill gaps by maintaining realistic backbone connectivity
    for i in range(len(query_coords)):
        if np.isnan(query_coords[i, 0]):
            # Find nearest valid neighbors
            prev_valid = next_valid = None
            
            for j in range(i-1, -1, -1):
                if not np.isnan(query_coords[j, 0]):
                    prev_valid = j
                    break
                    
            for j in range(i+1, len(query_coords)):
                if not np.isnan(query_coords[j, 0]):
                    next_valid = j
                    break
            
            if prev_valid is not None and next_valid is not None:
                # Interpolate along realistic RNA backbone path
                gap_size = next_valid - prev_valid
                total_distance = np.linalg.norm(query_coords[next_valid] - query_coords[prev_valid])
                expected_distance = gap_size * backbone_distance
                
                # If gap is compressed, extend it realistically
                if total_distance < expected_distance * 0.7:
                    direction = query_coords[next_valid] - query_coords[prev_valid]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                    
                    # Place intermediate points along extended path
                    for k, idx in enumerate(range(prev_valid + 1, next_valid)):
                        progress = (k + 1) / gap_size
                        base_pos = query_coords[prev_valid] + direction * expected_distance * progress
                        
                        # Add slight curvature for realism
                        perpendicular = np.cross(direction, [0, 0, 1])
                        if np.linalg.norm(perpendicular) < 1e-6:
                            perpendicular = np.cross(direction, [1, 0, 0])
                        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-10)
                        
                        curve_amplitude = 2.0 * np.sin(progress * np.pi)
                        query_coords[idx] = base_pos + perpendicular * curve_amplitude
                else:
                    # Linear interpolation for normal gaps
                    for k, idx in enumerate(range(prev_valid + 1, next_valid)):
                        weight = (k + 1) / gap_size
                        query_coords[idx] = (1 - weight) * query_coords[prev_valid] + weight * query_coords[next_valid]
            
            elif prev_valid is not None:
                # Extend from previous position
                if prev_valid > 0 and not np.isnan(query_coords[prev_valid-1, 0]):
                    direction = query_coords[prev_valid] - query_coords[prev_valid-1]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                else:
                    direction = np.array([1.0, 0.0, 0.0])
                
                steps_needed = i - prev_valid
                for step in range(1, steps_needed + 1):
                    pos_idx = prev_valid + step
                    if pos_idx < len(query_coords):
                        query_coords[pos_idx] = query_coords[prev_valid] + direction * backbone_distance * step
            
            elif next_valid is not None:
                # Work backwards from next position
                direction = np.array([-1.0, 0.0, 0.0])  # Default backward direction
                steps_needed = next_valid - i
                for step in range(steps_needed, 0, -1):
                    pos_idx = next_valid - step
                    if pos_idx >= 0:
                        query_coords[pos_idx] = query_coords[next_valid] - direction * backbone_distance * step
    
    # Final cleanup
    query_coords = np.nan_to_num(query_coords)
    return query_coords


def generate_improved_rna_structure(sequence):
    """
    Generate a more realistic RNA structure fallback based on sequence patterns
    and basic RNA structure principles.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        Array of 3D coordinates
    """
    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))
    
    # Analyze sequence to predict structural elements
    # Look for complementary regions that could form base pairs
    potential_stems = identify_potential_stems(sequence)
    
    # Default parameters
    radius_helix = 10.0
    radius_loop = 15.0
    rise_per_residue_helix = 2.5
    rise_per_residue_loop = 1.5
    angle_per_residue_helix = 0.6
    angle_per_residue_loop = 0.3
    
    # Assign structural classifications
    structure_types = assign_structure_types(sequence, potential_stems)
    
    # Generate coordinates based on predicted structure
    current_pos = np.array([0.0, 0.0, 0.0])
    current_direction = np.array([0.0, 0.0, 1.0])
    current_angle = 0.0
    
    for i in range(n_residues):
        if structure_types[i] == 'stem':
            # Part of a helical stem
            current_angle += angle_per_residue_helix
            coordinates[i] = [
                radius_helix * np.cos(current_angle), 
                radius_helix * np.sin(current_angle), 
                current_pos[2] + rise_per_residue_helix
            ]
            current_pos = coordinates[i]
        elif structure_types[i] == 'loop':
            # Part of a loop
            current_angle += angle_per_residue_loop
            z_shift = rise_per_residue_loop * np.sin(current_angle * 0.5)
            coordinates[i] = [
                radius_loop * np.cos(current_angle), 
                radius_loop * np.sin(current_angle), 
                current_pos[2] + z_shift
            ]
            current_pos = coordinates[i]
        else:
            # Single-stranded region
            # Add some randomness to make it look more realistic
            jitter = np.random.normal(0, 1, 3) * 2.0
            coordinates[i] = current_pos + jitter
            current_pos = coordinates[i]
            
    return coordinates

def identify_potential_stems(sequence):
    """
    Identify potential stem regions by looking for self-complementary segments.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        List of tuples (start1, end1, start2, end2) representing potentially paired regions
    """
    complementary_bases = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    min_stem_length = 3
    potential_stems = []
    
    # Simple stem identification
    for i in range(len(sequence) - min_stem_length):
        for j in range(i + min_stem_length + 3, len(sequence) - min_stem_length + 1):
            # Check if regions could form a stem
            potential_stem_len = min(min_stem_length, len(sequence) - j)
            is_stem = True
            
            for k in range(potential_stem_len):
                if sequence[i+k] not in complementary_bases or \
                   complementary_bases[sequence[i+k]] != sequence[j+potential_stem_len-k-1]:
                    is_stem = False
                    break
            
            if is_stem:
                potential_stems.append((i, i+potential_stem_len-1, j, j+potential_stem_len-1))
    
    return potential_stems

def assign_structure_types(sequence, potential_stems):
    """
    Assign each nucleotide to a structural element type.
    
    Args:
        sequence: RNA sequence string
        potential_stems: List of tuples representing stem regions
        
    Returns:
        List of structure types ('stem', 'loop', 'single')
    """
    structure_types = ['single'] * len(sequence)
    
    # Mark stem regions
    for stem in potential_stems:
        start1, end1, start2, end2 = stem
        for i in range(end1 - start1 + 1):
            structure_types[start1 + i] = 'stem'
            structure_types[end2 - i] = 'stem'
    
    # Mark loop regions (regions between paired regions)
    for i in range(len(potential_stems) - 1):
        _, end1, start2, _ = potential_stems[i]
        next_start1, _, _, _ = potential_stems[i+1]
        
        if next_start1 > end1 + 1 and start2 > next_start1:
            for j in range(end1 + 1, next_start1):
                structure_types[j] = 'loop'
    
    return structure_types


# Function to create a more realistic RNA structure when no good templates are found
def generate_rna_structure(sequence, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))
    
    # Initialize the first few residues in a helix
    for i in range(min(3, n_residues)):
        angle = i * 0.6
        coordinates[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]
    
    # Add more complex folding patterns
    current_direction = np.array([0.0, 0.0, 1.0])  # Start moving along z-axis
    
    # Define base-pairing tendencies (G-C and A-U pairs)
    for i in range(3, n_residues):
        # Check for potential base-pairing in the sequence
        has_pair = False
        pair_idx = -1
        
        # Simple detection of complementary bases (G-C, A-U)
        complementary = {'G': 'C', 'C': 'G', 'A': 'U', 'U': 'A'}
        current_base = sequence[i]
        
        # Look for potential base-pairing within a window before the current position
        window_size = min(i, 15)  # Look back up to 15 bases
        for j in range(i-window_size, i):
            if j >= 0 and sequence[j] == complementary.get(current_base, 'X'):
                # Found a potential pair
                has_pair = True
                pair_idx = j
                break
        
        if has_pair and i - pair_idx <= 10 and random.random() < 0.7:
            # Try to create a base-pair by positioning this nucleotide near its pair
            pair_pos = coordinates[pair_idx]
            
            # Create a position that's roughly opposite to the pair
            random_offset = np.random.normal(0, 1, 3) * 2.0
            base_pair_distance = 10.0 + random.uniform(-1.0, 1.0)
            
            # Calculate a vector from base-pair toward center of structure
            center = np.mean(coordinates[:i], axis=0)
            direction = center - pair_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Position new nucleotide in the general direction of the "center"
            coordinates[i] = pair_pos + direction * base_pair_distance + random_offset
            
            # Update direction for next nucleotide
            current_direction = np.random.normal(0, 0.3, 3)
            current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)
            
        else:
            # No base-pairing detected, continue with the current fold direction
            # Randomly rotate current direction to simulate RNA flexibility
            if random.random() < 0.3:
                # More significant direction change
                angle = random.uniform(0.2, 0.6)
                axis = np.random.normal(0, 1, 3)
                axis = axis / (np.linalg.norm(axis) + 1e-10)
                rotation = R.from_rotvec(angle * axis)
                current_direction = rotation.apply(current_direction)
            else:
                # Small random changes in direction
                current_direction += np.random.normal(0, 0.15, 3)
                current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)
            
            # Distance between consecutive nucleotides (3.5-4.5Å is typical)
            step_size = random.uniform(3.5, 4.5)
            
            # Update position
            coordinates[i] = coordinates[i-1] + step_size * current_direction
    
    return coordinates


def predict_rna_structures(sequence, target_id, train_seqs_df, train_coords_dict, n_predictions=5):
    predictions = []
    
    # Find similar sequences in the training data
    similar_seqs = find_similar_sequences(sequence, train_seqs_df, train_coords_dict, top_n=n_predictions)
    
    # If we found any similar sequences, use them as templates
    if similar_seqs:
        for i, (template_id, template_seq, similarity_score, template_coords) in enumerate(similar_seqs):
            # Adapt template coordinates to the query sequence
            adapted_coords = adapt_template_to_query(sequence, template_seq, template_coords)
            
            if adapted_coords is not None:
                # Apply adaptive constraints based on template similarity
                # For high similarity templates, apply very gentle constraints
                refined_coords = adaptive_rna_constraints(adapted_coords, sequence, confidence=similarity_score)
                
                # Add some randomness (less for better templates)
                random_scale = max(0.05, 0.8 - similarity_score)  # Reduced randomness
                randomized_coords = refined_coords.copy()
                randomized_coords += np.random.normal(0, random_scale, randomized_coords.shape)
                
                predictions.append(randomized_coords)
                
                if len(predictions) >= n_predictions:
                    break
    
    # If we don't have enough predictions from templates, generate de novo structures
    while len(predictions) < n_predictions:
        seed_value = hash(target_id) % 10000 + len(predictions) * 1000
        de_novo_coords = generate_rna_structure(sequence, seed=seed_value)
        
        # Apply stronger constraints to de novo structures (lower confidence)
        refined_de_novo = adaptive_rna_constraints(de_novo_coords, sequence, confidence=0.2)
        
        predictions.append(refined_de_novo)
    
    return predictions[:n_predictions]


# List to store all prediction records
all_predictions = []

# Set up time tracking
start_time = time.time()
total_targets = len(test_seqs)

# For each sequence in the test set
for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    
    # Progress tracking
    if idx % 5 == 0:
        elapsed = time.time() - start_time
        targets_processed = idx + 1
        if targets_processed > 0:
            avg_time_per_target = elapsed / targets_processed
            est_time_remaining = avg_time_per_target * (total_targets - targets_processed)
            print(f"Processing target {targets_processed}/{total_targets}: {target_id} ({len(sequence)} nt), "
                  f"elapsed: {elapsed:.1f}s, est. remaining: {est_time_remaining:.1f}s")
    
    # Generate 5 different structure predictions
    predictions = predict_rna_structures(sequence, target_id, train_seqs_extended, train_coords_dict, n_predictions=5)
    
    # For each residue in the sequence
    for j in range(len(sequence)):
        pred_row = {
            'ID': f"{target_id}_{j+1}",
            'resname': sequence[j],
            'resid': j + 1
        }
        
        # Add coordinates from all 5 predictions
        for i in range(5):
            pred_row[f'x_{i+1}'] = predictions[i][j][0]
            pred_row[f'y_{i+1}'] = predictions[i][j][1]
            pred_row[f'z_{i+1}'] = predictions[i][j][2]
        
        all_predictions.append(pred_row)

# Create DataFrame with predictions
submission_df = pd.DataFrame(all_predictions)

# Ensure the submission file has the correct format
column_order = ['ID', 'resname', 'resid']
for i in range(1, 6):
    for coord in ['x', 'y', 'z']:
        column_order.append(f'{coord}_{i}')
submission_df = submission_df[column_order]

# Save the submission file
submission_df.to_csv('submission.csv', index=False)
print(f"Generated predictions for {len(test_seqs)} RNA sequences")
print(f"Total runtime: {time.time() - start_time:.1f} seconds")


# Processing target 1/12: R1107 (69 nt), elapsed: 0.0s, est. remaining: 0.0s
# Processing target 6/12: R1128 (238 nt), elapsed: 122.8s, est. remaining: 122.8s
# Processing target 11/12: R1189 (118 nt), elapsed: 329.5s, est. remaining: 30.0s
# Generated predictions for 12 RNA sequences
# Total runtime: 421.8 seconds


submission_df

# 	ID	resname	resid	x_1	y_1	z_1	x_2	y_2	z_2	x_3	y_3	z_3	x_4	y_4	z_4	x_5	y_5	z_5
# 0	R1107_1	G	1	-5.553002	8.511589	8.570572	29.862446	27.622289	9.097237	-5.907881	-40.153475	-37.734814	-8.731645	19.134944	44.149403	17.251699	12.532484	80.377447
# 1	R1107_2	G	2	-5.769960	10.430263	13.913717	23.731256	26.004987	7.980069	-4.178720	-36.695692	-33.149475	-9.705379	24.302956	41.160891	12.236904	14.159156	77.481987
# 2	R1107_3	G	3	-5.834496	14.741914	17.565067	18.626321	24.355878	7.188991	-6.644076	-34.360592	-28.924777	-9.613496	27.980105	36.616509	9.213571	17.784742	74.452947
# 3	R1107_4	G	4	-5.708210	20.144298	18.735596	12.792807	23.097827	6.320889	-11.368209	-33.898344	-26.537797	-6.832251	30.008242	32.749573	9.010272	23.083631	72.576883
# 4	R1107_5	G	5	-5.727416	25.608371	17.127092	7.377047	20.380882	5.038376	-15.399877	-36.019782	-22.904240	-3.491603	28.517940	29.050341	11.735484	27.602889	70.667134
# ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
# 2510	R1190_114	U	114	84.059142	106.137124	76.417536	139.984054	173.011385	35.489986	53.019749	5.864413	41.011041	-2.521784	-46.131563	30.060610	12.471272	171.367045	72.491686
# 2511	R1190_115	U	115	83.655243	100.664127	76.796596	143.146568	177.405424	37.265375	52.750846	11.022034	40.027335	-3.489576	-50.020212	33.526826	11.805377	169.128232	67.808148
# 2512	R1190_116	U	116	83.338063	95.378380	75.327400	146.282348	182.526123	38.399489	50.417980	5.518266	41.454176	-3.738666	-54.037065	38.283867	9.351742	165.609850	64.584470
# 2513	R1190_117	U	117	85.467708	91.282386	71.790222	149.131272	187.397092	39.962140	52.286340	10.636763	38.093230	-3.985077	-57.967752	42.167757	3.726637	164.021770	63.228448
# 2514	R1190_118	U	118	92.292804	85.926956	72.708491	152.775057	192.448426	41.259409	55.699243	12.921680	33.804887	-5.141289	-62.061082	46.862255	0.588024	162.089915	56.700059

# 2515 rows × 18 columns

