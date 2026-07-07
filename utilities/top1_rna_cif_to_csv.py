# heyy
!pip install -q /kaggle/input/rna-wheels/wheels/biopython-1.85-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!ls "/kaggle/input/stanford-rna-3d-folding"
MSA		       test_sequences.csv   train_sequences.v2.csv
MSA_v2		       train_labels.csv     validation_labels.csv
PDB_RNA		       train_labels.v2.csv  validation_sequences.csv
sample_submission.csv  train_sequences.csv
import os
from tqdm import tqdm
from collections import Counter

files = os.listdir("/kaggle/input/stanford-rna-3d-folding/PDB_RNA")
print(f"Total number of files: {len(files)}")

# Get file extensions
extensions = [os.path.splitext(file)[1] for file in files if os.path.splitext(file)[1]]

# Count occurrences of each extension
extension_counts = Counter(extensions)

print(f"File types and their counts:")
for ext, count in sorted(extension_counts.items()):
    print(f"  {ext}: {count}")
Total number of files: 8672
File types and their counts:
  .cif: 8670
  .csv: 1
  .fasta: 1
import pandas as pd
import os

# Find the CSV file name
files = os.listdir("/kaggle/input/stanford-rna-3d-folding/PDB_RNA")
csv_files = [f for f in files if f.endswith('.csv')]
print(f"CSV file(s): {csv_files}")

# Read the CSV file
csv_file_path = f"/kaggle/input/stanford-rna-3d-folding/PDB_RNA/{csv_files[0]}"

# Skip the problematic header lines
df = pd.read_csv(csv_file_path, on_bad_lines='skip')
print(f"CSV file shape: {df.shape}")

print(f"Column names: {list(df.columns)}")
print(df.head())
CSV file(s): ['pdb_release_dates_NA.csv']
CSV file shape: (8679, 2)
Column names: ['Entry ID', 'Release Date']
  Entry ID Release Date
0     124D   1993-10-31
1     157D   1994-05-31
2     165D   1994-08-31
3     176D   1994-11-01
4     17RA   1999-04-20
# Check unique counts for Entry ID
print(f"Total rows: {len(df)}")
print(f"Unique Entry IDs: {df['Entry ID'].nunique()}")
print(f"Duplicate Entry IDs: {len(df) - df['Entry ID'].nunique()}")

# Convert Release Date to datetime for better analysis
df['Release Date'] = pd.to_datetime(df['Release Date'])

# Get date range
print(f"\nRelease Date range:")
print(f"Earliest date: {df['Release Date'].min()}")
print(f"Latest date: {df['Release Date'].max()}")
print(f"Date span: {(df['Release Date'].max() - df['Release Date'].min()).days} days")

# Check unique release dates
print(f"\nUnique Release Dates: {df['Release Date'].nunique()}")

# Show some examples of duplicates if any exist
if len(df) > df['Entry ID'].nunique():
    print(f"\nExamples of duplicate Entry IDs:")
    duplicates = df[df['Entry ID'].duplicated(keep=False)].sort_values('Entry ID')
    print(duplicates.head(10))
Total rows: 8679
Unique Entry IDs: 8679
Duplicate Entry IDs: 0

Release Date range:
Earliest date: 1978-04-12 00:00:00
Latest date: 2025-05-07 00:00:00
Date span: 17192 days

Unique Release Dates: 1339
 
FASTA:
import os

# Find the FASTA file
files = os.listdir("/kaggle/input/stanford-rna-3d-folding/PDB_RNA")
fasta_files = [f for f in files if f.endswith('.fasta')]
print(f"FASTA file(s): {fasta_files}")

# Read and examine the FASTA file
fasta_file_path = f"/kaggle/input/stanford-rna-3d-folding/PDB_RNA/{fasta_files[0]}"

# Read the file and look at its structure
with open(fasta_file_path, 'r') as f:
    lines = f.readlines()

print(f"Total lines in FASTA file: {len(lines)}")
print(f"\nFirst 20 lines:")
for i, line in enumerate(lines[:20]):
    print(f"Line {i+1}: {repr(line)}")
FASTA file(s): ['pdb_seqres_NA.fasta']
Total lines in FASTA file: 135350

First 20 lines:
Line 1: ">100d_A mol:na length:10  DNA/RNA (5'-R(*CP*)-D(*CP*GP*GP*CP*GP*CP*CP*GP*)-R(*G)-3')\n"
Line 2: 'CCGGCGCCGG\n'
Line 3: ">100d_B mol:na length:10  DNA/RNA (5'-R(*CP*)-D(*CP*GP*GP*CP*GP*CP*CP*GP*)-R(*G)-3')\n"
Line 4: 'CCGGCGCCGG\n'
Line 5: ">101d_A mol:na length:12  DNA (5'-D(*CP*GP*CP*GP*AP*AP*TP*TP*(CBR)P*GP*CP*G)-3')\n"
Line 6: 'CGCGAATTCGCG\n'
Line 7: ">101d_B mol:na length:12  DNA (5'-D(*CP*GP*CP*GP*AP*AP*TP*TP*(CBR)P*GP*CP*G)-3')\n"
Line 8: 'CGCGAATTCGCG\n'
Line 9: '--\n'
Line 10: ">102d_A mol:na length:12  DNA (5'-D(*CP*GP*CP*AP*AP*AP*TP*TP*TP*GP*CP*G)-3')\n"
Line 11: 'CGCAAATTTGCG\n'
Line 12: ">102d_B mol:na length:12  DNA (5'-D(*CP*GP*CP*AP*AP*AP*TP*TP*TP*GP*CP*G)-3')\n"
Line 13: 'CGCAAATTTGCG\n'
Line 14: '--\n'
Line 15: ">103d_A mol:na length:12  DNA (5'-D(*GP*TP*GP*GP*AP*AP*TP*GP*GP*AP*AP*C)-3')\n"
Line 16: 'GTGGAATGGAAC\n'
Line 17: ">103d_B mol:na length:12  DNA (5'-D(*GP*TP*GP*GP*AP*AP*TP*GP*GP*AP*AP*C)-3')\n"
Line 18: 'GTGGAATGGAAC\n'
Line 19: '--\n'
Line 20: ">104d_A mol:na length:12  DNA/RNA (5'-R(*CP*GP*CP*G)-D(P*TP*AP*TP*AP*CP*GP*CP*G)-3')\n"
# Parse FASTA file to count sequences
sequences = []
current_seq = ""
headers = []

with open(fasta_file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):  # Header line
            if current_seq:  # Save previous sequence
                sequences.append(current_seq)
                current_seq = ""
            headers.append(line)
        else:  # Sequence line
            current_seq += line
    
    # Don't forget the last sequence
    if current_seq:
        sequences.append(current_seq)

print(f"\nFASTA file summary:")
print(f"Number of sequences: {len(sequences)}")
print(f"Number of headers: {len(headers)}")

if sequences:
    seq_lengths = [len(seq) for seq in sequences]
    print(f"Sequence lengths - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Average: {sum(seq_lengths)/len(seq_lengths):.1f}")
    
    print(f"\nFirst few headers:")
    for i, header in enumerate(headers[:5]):
        print(f"  {header}")
    
    print(f"\nFirst sequence (first 100 chars):")
    print(f"  {sequences[0][:100]}...")
FASTA file summary:
Number of sequences: 56501
Number of headers: 56501
Sequence lengths - Min: 1, Max: 16183, Average: 231.4

First few headers:
  >100d_A mol:na length:10  DNA/RNA (5'-R(*CP*)-D(*CP*GP*GP*CP*GP*CP*CP*GP*)-R(*G)-3')
  >100d_B mol:na length:10  DNA/RNA (5'-R(*CP*)-D(*CP*GP*GP*CP*GP*CP*CP*GP*)-R(*G)-3')
  >101d_A mol:na length:12  DNA (5'-D(*CP*GP*CP*GP*AP*AP*TP*TP*(CBR)P*GP*CP*G)-3')
  >101d_B mol:na length:12  DNA (5'-D(*CP*GP*CP*GP*AP*AP*TP*TP*(CBR)P*GP*CP*G)-3')
  >102d_A mol:na length:12  DNA (5'-D(*CP*GP*CP*AP*AP*AP*TP*TP*TP*GP*CP*G)-3')

First sequence (first 100 chars):
  CCGGCGCCGG...
# Continue with the parsing code from before to see the complete statistics
print(f"\nFASTA file summary:")
print(f"Number of sequences: {len(sequences)}")
print(f"Sequence lengths - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}")

# Count DNA vs RNA vs mixed
dna_count = sum(1 for h in headers if 'DNA' in h and 'RNA' not in h)
rna_count = sum(1 for h in headers if 'RNA' in h and 'DNA' not in h)
mixed_count = sum(1 for h in headers if 'DNA' in h and 'RNA' in h)
print(f"DNA sequences: {dna_count}")
print(f"RNA sequences: {rna_count}")  
print(f"Mixed DNA/RNA: {mixed_count}")

# Count unique PDB IDs
pdb_ids = [h.split('_')[0][1:] for h in headers]  # Remove '>' and chain part
unique_pdbs = len(set(pdb_ids))
print(f"Unique PDB structures: {unique_pdbs}")
FASTA file summary:
Number of sequences: 56501
Sequence lengths - Min: 1, Max: 16183
DNA sequences: 23722
RNA sequences: 20746
Mixed DNA/RNA: 455
Unique PDB structures: 19422
# Count RNA sequences with less than 1000 nucleotides
rna_short_count = 0
rna_lengths = []

for i, header in enumerate(headers):
   if 'RNA' in header and 'DNA' not in header:  # Pure RNA sequences only
       seq_length = len(sequences[i])
       rna_lengths.append(seq_length)
       if seq_length < 1000:
           rna_short_count += 1

print(f"RNA sequences with less than 1000 nucleotides: {rna_short_count}")
print(f"Total RNA sequences: {len(rna_lengths)}")
print(f"Percentage of RNA sequences < 1000 nt: {rna_short_count/len(rna_lengths)*100:.1f}%")

# Additional statistics for RNA sequences
if rna_lengths:
   print(f"\nRNA sequence length statistics:")
   print(f"Min length: {min(rna_lengths)}")
   print(f"Max length: {max(rna_lengths)}")
   print(f"Average length: {sum(rna_lengths)/len(rna_lengths):.1f}")
   
   # Length distribution
   length_ranges = [
       (0, 100, "1-100"),
       (100, 500, "100-500"), 
       (500, 1000, "500-1000"),
       (1000, 2000, "1000-2000"),
       (2000, float('inf'), "2000+")
   ]
   
   print(f"\nRNA sequence length distribution:")
   for min_len, max_len, label in length_ranges:
       count = sum(1 for length in rna_lengths if min_len < length <= max_len)
       print(f"  {label} nt: {count}")
RNA sequences with less than 1000 nucleotides: 16344
Total RNA sequences: 20746
Percentage of RNA sequences < 1000 nt: 78.8%

RNA sequence length statistics:
Min length: 2
Max length: 16183
Average length: 564.2

RNA sequence length distribution:
  1-100 nt: 12448
  100-500 nt: 3680
  500-1000 nt: 216
  1000-2000 nt: 2224
  2000+ nt: 2178
Train Sequences:
train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_sequences.head()
target_id	sequence	temporal_cutoff	description	all_sequences
0	1SCL_A	GGGUGCUCAGUACGAGAGGAACCGCACCC	1995-01-26	THE SARCIN-RICIN LOOP, A MODULAR RNA	>1SCL_1|Chain A|RNA SARCIN-RICIN LOOP|Rattus n...
1	1RNK_A	GGCGCAGUGGGCUAGCGCCACUCAAAAGGCCCAU	1995-02-27	THE STRUCTURE OF AN RNA PSEUDOKNOT THAT CAUSES...	>1RNK_1|Chain A|RNA PSEUDOKNOT|null\nGGCGCAGUG...
2	1RHT_A	GGGACUGACGAUCACGCAGUCUAU	1995-06-03	24-MER RNA HAIRPIN COAT PROTEIN BINDING SITE F...	>1RHT_1|Chain A|RNA (5'-R(P*GP*GP*GP*AP*CP*UP*...
3	1HLX_A	GGGAUAACUUCGGUUGUCCC	1995-09-15	P1 HELIX NUCLEIC ACIDS (DNA/RNA) RIBONUCLEIC ACID	>1HLX_1|Chain A|RNA (5'-R(*GP*GP*GP*AP*UP*AP*A...
4	1HMH_E	GGCGACCCUGAUGAGGCCGAAAGGCCGAAACCGU	1995-12-07	THREE-DIMENSIONAL STRUCTURE OF A HAMMERHEAD RI...	>1HMH_1|Chains A, C, E|HAMMERHEAD RIBOZYME-RNA...
train_sequences_v2 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.v2.csv")
train_sequences_v2.head()
target_id	sequence	temporal_cutoff	description	all_sequences
0	7TAX_M	CUAAGAAAUUCACGGCGGGCUUGAUGUCCGCGUCUACCUGAUUCAC...	2022-09-21	Cryo-EM structure of the Csy-AcrIF24-promoter ...	>7TAX_1|Chain A|CRISPR-associated protein Csy1...
1	4WF1_CA	AAUUGAAGAGUUUGAUCAUGGCUCAGAUUGAACGCUGGCGGCAGGC...	2014-11-05	Crystal structure of the E. coli ribosome boun...	>4WF1_1|Chains A[auth AA], BB[auth CA]|16S rRN...
2	8PVA_b	UGCCUGGCGGCCGUAGCGCGGUGGUCCCACCUGACCCCAUGCCGAA...	2023-11-29	Structure of bacterial ribosome determined by ...	>8PVA_1|Chain A|16S rRNA|Escherichia coli (562...
3	8OVE_BB	CAACUGCAGACCGUACUCAUCACCGCAUCAGGUCCCCAAGCAUCGA...	2023-11-29	CRYO-EM STRUCTURE OF TRYPANOSOMA BRUCEI PROCYC...	>8OVE_1|Chain A[auth AA]|SSU rRNA|Trypanosoma ...
4	8JDL_w	UACCUGGUUGAUCCUGCCAGUAGCAUUGCUUGCCAAAGAUUAAGCC...	2023-12-06	Structure of the Human cytoplasmic Ribosome wi...	>8JDL_1|Chain A|mRNA|Homo sapiens (9606)\nUUAU...
# Check for target_ids in v1 that are not in v2
v1_only_targets = set(train_sequences['target_id']) - set(train_sequences_v2['target_id'])
print(f"Target IDs in v1 but not in v2: {len(v1_only_targets)}")
print(f"First 10: {list(v1_only_targets)[:10]}")
Target IDs in v1 but not in v2: 244
First 10: ['1BAU_B', '3W1K_F', '6BZ7_QW', '4V5Z_BC', '1X18_B', '6KUT_V', '6XWJ_A', '1LS2_B', '1QWA_A', '8SFO_B']
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
train_labels_v2 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.v2.csv")
train_labels.head()
ID	resname	resid	x_1	y_1	z_1
0	1SCL_A_1	G	1	13.760	-25.974001	0.102
1	1SCL_A_2	G	2	9.310	-29.638000	2.669
2	1SCL_A_3	G	3	5.529	-27.813000	5.878
3	1SCL_A_4	U	4	2.678	-24.900999	9.793
4	1SCL_A_5	G	5	1.827	-20.136000	11.793
train_labels_v2.head()
ID	resname	resid	x_1	y_1	z_1
0	7TAX_M_1	C	1	187.126007	148.246002	210.417999
1	7TAX_M_2	U	2	185.255997	152.968002	204.617996
2	7TAX_M_3	A	3	189.360992	161.802002	205.214996
3	7TAX_M_4	A	4	186.000000	156.595993	209.951996
4	7TAX_M_5	G	5	181.947998	158.186996	213.610992
# Extract target_ids from labels by removing the residue suffix
v1_label_targets = set(train_labels['ID'].str.rsplit('_', n=1).str[0])
v2_label_targets = set(train_labels_v2['ID'].str.rsplit('_', n=1).str[0])

# Check for target_ids in v1 labels that are not in v2 labels
v1_only_label_targets = v1_label_targets - v2_label_targets
print(f"Target IDs in v1 labels but not in v2 labels: {len(v1_only_label_targets)}")
print(f"First 10: {list(v1_only_label_targets)[:10]}")
Target IDs in v1 labels but not in v2 labels: 244
First 10: ['1BAU_B', '3W1K_F', '6BZ7_QW', '4V5Z_BC', '1X18_B', '6KUT_V', '6XWJ_A', '1LS2_B', '1QWA_A', '8SFO_B']
# Concatenate both sequence dataframes
combined_sequences = pd.concat([train_sequences, train_sequences_v2], ignore_index=True)

# Concatenate both label dataframes  
combined_labels = pd.concat([train_labels, train_labels_v2], ignore_index=True)

print(f"Combined sequences shape: {combined_sequences.shape}")
print(f"Combined labels shape: {combined_labels.shape}")
print(f"Unique target_ids in combined sequences: {combined_sequences['target_id'].nunique()}")
Combined sequences shape: (5979, 5)
Combined labels shape: (3814190, 6)
Unique target_ids in combined sequences: 5379
combined_sequences.head()
target_id	sequence	temporal_cutoff	description	all_sequences
0	1SCL_A	GGGUGCUCAGUACGAGAGGAACCGCACCC	1995-01-26	THE SARCIN-RICIN LOOP, A MODULAR RNA	>1SCL_1|Chain A|RNA SARCIN-RICIN LOOP|Rattus n...
1	1RNK_A	GGCGCAGUGGGCUAGCGCCACUCAAAAGGCCCAU	1995-02-27	THE STRUCTURE OF AN RNA PSEUDOKNOT THAT CAUSES...	>1RNK_1|Chain A|RNA PSEUDOKNOT|null\nGGCGCAGUG...
2	1RHT_A	GGGACUGACGAUCACGCAGUCUAU	1995-06-03	24-MER RNA HAIRPIN COAT PROTEIN BINDING SITE F...	>1RHT_1|Chain A|RNA (5'-R(P*GP*GP*GP*AP*CP*UP*...
3	1HLX_A	GGGAUAACUUCGGUUGUCCC	1995-09-15	P1 HELIX NUCLEIC ACIDS (DNA/RNA) RIBONUCLEIC ACID	>1HLX_1|Chain A|RNA (5'-R(*GP*GP*GP*AP*UP*AP*A...
4	1HMH_E	GGCGACCCUGAUGAGGCCGAAAGGCCGAAACCGU	1995-12-07	THREE-DIMENSIONAL STRUCTURE OF A HAMMERHEAD RI...	>1HMH_1|Chains A, C, E|HAMMERHEAD RIBOZYME-RNA...
# Extract target_ids from FASTA headers for RNA sequences only
rna_headers = [h for h in headers if 'RNA' in h and 'DNA' not in h]
fasta_rna_targets = set()

for header in rna_headers:
    # Extract PDB_ID and chain from header like ">1SCL_A mol:na..."
    target_part = header.split()[0][1:]  # Remove '>' and take first part
    fasta_rna_targets.add(target_part)

print(f"RNA targets in FASTA: {len(fasta_rna_targets)}")

# Check overlap with combined sequences
combined_targets = set(combined_sequences['target_id'])
overlap = combined_targets.intersection(fasta_rna_targets)
missing_in_fasta = combined_targets - fasta_rna_targets

print(f"Combined sequence targets: {len(combined_targets)}")
print(f"Overlap (targets in both): {len(overlap)}")
print(f"Missing in FASTA: {len(missing_in_fasta)}")
print(f"Coverage: {len(overlap)/len(combined_targets)*100:.1f}%")
RNA targets in FASTA: 20746
Combined sequence targets: 5379
Overlap (targets in both): 0
Missing in FASTA: 5379
Coverage: 0.0%
# Check unique target_id counts in combined sequences
target_counts = combined_sequences['target_id'].value_counts()
print(f"Total unique target_ids: {len(target_counts)}")
print(f"Target_ids appearing more than once: {(target_counts > 1).sum()}")
print(f"\nFirst 10 target_ids and their counts:")
print(target_counts.head(10))

print(f"\nSample target_ids from combined_sequences:")
print(combined_sequences['target_id'].head(10).tolist())

print(f"\nSample target_ids from FASTA RNA headers:")
sample_fasta_targets = list(fasta_rna_targets)[:10]
print(sample_fasta_targets)
Total unique target_ids: 5379
Target_ids appearing more than once: 600

First 10 target_ids and their counts:
target_id
1SCL_A    2
5IEM_A    2
2NC1_A    2
2NC0_A    2
5KK5_B    2
2NBZ_A    2
2NBX_A    2
5KQE_A    2
2NCI_A    2
5KMZ_A    2
Name: count, dtype: int64

Sample target_ids from combined_sequences:
['1SCL_A', '1RNK_A', '1RHT_A', '1HLX_A', '1HMH_E', '1RNG_A', '1MME_D', '1KAJ_A', '1SLO_A', '1BIV_A']

Sample target_ids from FASTA RNA headers:
['6rt6_A', '4lnt_YA', '4v9c_CV', '5it7_8', '3adc_C', '8ekb_1x', '6rja_G', '3eph_E', '6ifk_N', '6mtb_7']
# Compare the naming patterns more clearly
print("Combined sequences target_id pattern:")
print("Format appears to be: [4-char PDB]_[1-2 char chain]")
for target in combined_sequences['target_id'].head(5):
    print(f"  {target}")

print(f"\nFASTA RNA target_id pattern:")
print("Format appears to be: [4-char PDB]_[1-2 char chain]")
for target in list(fasta_rna_targets)[:5]:
    print(f"  {target}")

# Check if there's any case sensitivity or format difference
combined_lower = set(t.lower() for t in combined_sequences['target_id'])
fasta_lower = set(t.lower() for t in fasta_rna_targets)
overlap_lower = combined_lower.intersection(fasta_lower)

print(f"\nCase-insensitive check:")
print(f"Overlap when ignoring case: {len(overlap_lower)}")

# Check PDB codes only (without chain)
combined_pdbs = set(t.split('_')[0] for t in combined_sequences['target_id'])
fasta_pdbs = set(t.split('_')[0] for t in fasta_rna_targets)
pdb_overlap = combined_pdbs.intersection(fasta_pdbs)

print(f"\nPDB code overlap (ignoring chains):")
print(f"Combined PDB codes: {len(combined_pdbs)}")
print(f"FASTA PDB codes: {len(fasta_pdbs)}")
print(f"PDB overlap: {len(pdb_overlap)}")
Combined sequences target_id pattern:
Format appears to be: [4-char PDB]_[1-2 char chain]
  1SCL_A
  1RNK_A
  1RHT_A
  1HLX_A
  1HMH_E

FASTA RNA target_id pattern:
Format appears to be: [4-char PDB]_[1-2 char chain]
  6rt6_A
  4lnt_YA
  4v9c_CV
  5it7_8
  3adc_C

Case-insensitive check:
Overlap when ignoring case: 4431

PDB code overlap (ignoring chains):
Combined PDB codes: 3932
FASTA PDB codes: 7354
PDB overlap: 0
# The case-insensitive overlap suggests the issue is case sensitivity
# Let's examine this more closely

print("Case comparison examples:")
combined_sample = list(combined_sequences['target_id'])[:5]
fasta_sample = list(fasta_rna_targets)[:5]

for target in combined_sample:
    print(f"Combined: {target} -> lowercase: {target.lower()}")

for target in fasta_sample:
    print(f"FASTA: {target} -> lowercase: {target.lower()}")

# Check if FASTA uses lowercase PDB codes
print(f"\nPDB code case analysis:")
combined_pdb_sample = [t.split('_')[0] for t in combined_sample]
fasta_pdb_sample = [t.split('_')[0] for t in fasta_sample]

print("Combined PDB codes:", combined_pdb_sample)
print("FASTA PDB codes:", fasta_pdb_sample)

# Check PDB overlap with case-insensitive comparison
combined_pdbs_lower = set(t.split('_')[0].lower() for t in combined_sequences['target_id'])
fasta_pdbs_lower = set(t.split('_')[0].lower() for t in fasta_rna_targets)
pdb_overlap_lower = combined_pdbs_lower.intersection(fasta_pdbs_lower)

print(f"\nCase-insensitive PDB overlap: {len(pdb_overlap_lower)}")
print(f"This explains the discrepancy - FASTA uses lowercase PDB codes!")
Case comparison examples:
Combined: 1SCL_A -> lowercase: 1scl_a
Combined: 1RNK_A -> lowercase: 1rnk_a
Combined: 1RHT_A -> lowercase: 1rht_a
Combined: 1HLX_A -> lowercase: 1hlx_a
Combined: 1HMH_E -> lowercase: 1hmh_e
FASTA: 6rt6_A -> lowercase: 6rt6_a
FASTA: 4lnt_YA -> lowercase: 4lnt_ya
FASTA: 4v9c_CV -> lowercase: 4v9c_cv
FASTA: 5it7_8 -> lowercase: 5it7_8
FASTA: 3adc_C -> lowercase: 3adc_c

PDB code case analysis:
Combined PDB codes: ['1SCL', '1RNK', '1RHT', '1HLX', '1HMH']
FASTA PDB codes: ['6rt6', '4lnt', '4v9c', '5it7', '3adc']

Case-insensitive PDB overlap: 3285
This explains the discrepancy - FASTA uses lowercase PDB codes!
cif files
# !ls /kaggle/input/stanford-rna-3d-folding/PDB_RNA
# Extract RNA sequences and coordinates with deduplication
from Bio.PDB import MMCIFParser
import pandas as pd
from pathlib import Path

def extract_rna_data_from_cif(cif_file_path):
    """Extract unique RNA sequences and C1' coordinates from a CIF file"""
    parser = MMCIFParser(QUIET=True)
    
    try:
        structure = parser.get_structure('structure', cif_file_path)
        pdb_id = Path(cif_file_path).stem.upper()
        
        sequences_data = []
        coordinates_data = []
        seen_sequences = set()  # Track unique sequences
        
        for model in structure:
            for chain in model:
                chain_id = chain.id
                target_id = f"{pdb_id}_{chain_id}"
                
                # Check if chain contains RNA residues
                rna_residues = []
                for residue in chain:
                    if residue.get_resname() in ['A', 'U', 'G', 'C']:  # RNA nucleotides
                        rna_residues.append(residue)
                
                if rna_residues:  # Only process if RNA residues found
                    # Build sequence
                    sequence = ''.join([res.get_resname() for res in rna_residues])
                    
                    # Only add if sequence is unique
                    if sequence not in seen_sequences:
                        seen_sequences.add(sequence)
                        sequences_data.append({
                            'target_id': target_id,
                            'sequence': sequence
                        })
                        
                        # Extract C1' coordinates for this unique sequence
                        for i, residue in enumerate(rna_residues, 1):
                            if "C1'" in residue:
                                atom = residue["C1'"]
                                coordinates_data.append({
                                    'ID': f"{target_id}_{i}",
                                    'resname': residue.get_resname(),
                                    'resid': i,
                                    'x_1': atom.coord[0],
                                    'y_1': atom.coord[1], 
                                    'z_1': atom.coord[2]
                                })
        
        return sequences_data, coordinates_data
        
    except Exception as e:
        print(f"Error processing {cif_file_path}: {e}")
        return [], []
# Process all CIF files and save to CSV
import os
from tqdm import tqdm

def process_all_cif_files():
    """Process all CIF files in the directory and extract RNA data"""
    cif_dir = "/kaggle/input/stanford-rna-3d-folding/PDB_RNA"
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    
    all_sequences = []
    all_coordinates = []
    
    print(f"Processing {len(cif_files)} CIF files...")
    
    for cif_file in tqdm(cif_files):
        cif_path = os.path.join(cif_dir, cif_file)
        sequences, coordinates = extract_rna_data_from_cif(cif_path)
        
        all_sequences.extend(sequences)
        all_coordinates.extend(coordinates)
    
    return all_sequences, all_coordinates

# Process all files
print("Starting full extraction...")
all_sequences, all_coordinates = process_all_cif_files()

print(f"\nFull extraction summary:")
print(f"Total unique RNA sequences: {len(all_sequences)}")
print(f"Total coordinate entries: {len(all_coordinates)}")

# Create DataFrames
sequences_df = pd.DataFrame(all_sequences)
coordinates_df = pd.DataFrame(all_coordinates)

# Save to CSV files
sequences_df.to_csv('rna_sequences.csv', index=False)
coordinates_df.to_csv('rna_coordinates.csv', index=False)

print(f"\nSaved files:")
print(f"rna_sequences.csv: {sequences_df.shape}")
print(f"rna_coordinates.csv: {coordinates_df.shape}")

print(f"\nFirst few entries:")
print(sequences_df.head())
Starting full extraction...
Processing 8670 CIF files...
100%|██████████| 8670/8670 [8:01:50<00:00,  3.33s/it]
Full extraction summary:
Total unique RNA sequences: 18881
Total coordinate entries: 10135546

Saved files:
rna_sequences.csv: (18881, 2)
rna_coordinates.csv: (10135546, 6)

First few entries:
  target_id                                           sequence
0    2D19_A                                  GCUGAAGUGCACACGGC
1   6OXI_QA  GUUGGAGAGUUUGAUCCUGGCUCAGGGUGAACGCUGGCGGCGUGCC...
2   6OXI_QV  CGCGGGGUGGAGCAGCCUGGUAGCUCGUCGGGCUCAUAACCCGAAG...
3   6OXI_QX                                  CAAGGAGGUAAAAAUGU
4   6OXI_RA  AGAUGGUAAGGGCCCACGGUGGAUGCCUCGGCACCCGAGCCGAUGA...
# MSA     - 856
# MSA_v2  - 2534
# PDB_RNA - 8672