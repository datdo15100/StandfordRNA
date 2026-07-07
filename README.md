Overview
If you sat down to complete a puzzle without knowing what it should look like, you’d have to rely on patterns and logic to piece it together. In the same way, predicting Ribonucleic acid (RNA)’s 3D structure involves using only its sequence to figure out how it folds into the structures that define its function.

In this competition, you’ll develop machine learning models to predict an RNA molecule’s 3D structure from its sequence. The goal is to improve our understanding of biological processes and drive new advancements in medicine and biotechnology.

Start

Feb 28, 2025
Close

Sep 25, 2025
Merger & Entry
Description
RNA is vital to life’s most essential processes, but despite its significance, predicting its 3D structure is still difficult. Deep learning breakthroughs like AlphaFold have transformed protein structure prediction, but progress with RNA has been much slower due to limited data and evaluation methods.

This competition builds on recent advances, like the deep learning foundation model RibonanzaNet, which emerged from a prior Kaggle competition. Now, you’ll take on the next challenge—predicting RNA’s full 3D structure.

Your work could push RNA-based medicine forward, making treatments like cancer immunotherapies and CRISPR gene editing more accessible and effective. More fundamentally, your work may be the key step in illuminating the folds and functions of natural RNA molecules, which have been called the 'dark matter of biology'.

This competition is made possible through a worldwide collaborative effort including the organizers, experimental RNA structural biologists, and predictors of the CASP16 and RNA-Puzzles competitions; Howard Hughes Medical Institute; the Institute of Protein Design; and Stanford University School of Medicine.

Evaluation
Submissions are scored using TM-score ("template modeling" score), which goes from 0.0 to 1.0 (higher is better):

The scoring code is ribonaza_tm_score.py
where:

Lref is the number of residues solved in the experimental reference structure ("ground truth").

Lalign is the number of aligned residues.

di is the distance between the ith pair of aligned residues, in Angstroms.

d0 is a distance scaling factor in Angstroms, defined as:


for Lref ≥ 30; and d0 = 0.3, 0.4, 0.5, 0.6, or 0.7 for Lref <12, 12-15, 16-19, 20-23, or 24-29, respectively.

The rotation and translation of predicted structures to align with experimental reference structures are carried out by US-align. To match default settings, as used in the CASP competitions, the alignment will be sequence-independent.

For each target RNA sequence, you will submit 5 predictions and your final score will be the average of best-of-5 TM-scores of all targets. For a few targets, multiple slightly different structures have been captured experimentally; your predictions' scores will be based on the best TM-score compared to each of these reference structures.

Submission File
For each sequence in the test set, you can predict five structures. Your notebook should look for a file test_sequences.csv and output submission.csv. This file should contain x, y, z coordinates of the C1' atom in each residue across your predicted structures 1 to 5:

ID,resname,resid,x_1,y_1,z_1,... x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,... -7.301,9.023,8.932
R1107_2,G,1,-8.02,11.014,14.606,... -7.953,10.02,12.127
etc.
You must submit five sets of coordinates.

Timeline
February 27, 2025 - Start Date.
April 23, 2025 - Public leaderboard refresh & Early Sharing Prizes
May 22, 2025 - Entry Deadline. You must accept the competition rules before this date in order to compete.
May 22, 2025 - Team Merger Deadline. This is the last day participants may join or merge teams.
May 29, 2025 - Final submissions deadline.
September 24, 2025 - Competition End Date.
All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Future Data Evaluation Timeline:
After the final submission deadline there will be periodic updates to the leaderboard to reflect up to 40 new RNA (sequences) generated after the competition has ended. New data updates that will be run against selected notebooks.

September 24, 2025 - Competition End Date - This date is subject to change based upon the availability of new sequences. Watch the forum after the competition end for updates.
Prizes
Leaderboard Prizes
1st Place - $ 45,000
2nd Place - $ 15,000
3rd Place - $ 10,000
Early Sharing Prizes
Participants of this competition are encouraged to make publicly available their notebooks through the competition. There will be a refresh of the public leaderboard 2 months after competition start. At that time, $2,500 will be awarded to the first two teams to publish a public notebook scoring above the VFOLD_human_expert score on the leaderboard. A discussion post will detail timing of the refresh.

To be eligible for the Early Sharing Prize, you will need to:

1) Publish a public notebook scoring above the benchmark score on the leaderboard after the data refresh (first two notebooks that meet this criteria will be evaluated).

2) Out of all participants or Teams who have submitted notebooks scoring above the benchmark score, be the first two to make your notebooks public. The public notebook needs to adhere to the same requirements and restrictions regarding licensing, reproducibility, and documentation to which the winning Submission is subject (see Competition Rules).

3) Keep the notebooks and any datasets they use publicly available until the Final Submission Deadline of May 29, 2025. Submissions should only make use of information publicly available before the temporal_cutoff dates provided with test sequences.

The Competition Sponsor will, after the data refresh, assess all Submissions that are eligible for the Early Sharing Prize in the order in which Submissions were made. If it is discovered that such a Submissions that scored more than the benchmark score has no or incomplete documentation, incompatible licensing, or is in any other way incompatible with the rules to which the winning Submission is subject, it will not be considered towards the Early Sharing Prize and the next Submissions will be assessed.

Paper Authorship
Top performing participants on the Public Leaderboard rankings at the final submission deadline will be invited to contribute their code and model descriptions to a scientific paper summarizing the competition's scientific outcome.

Code Requirements


This is a Code Competition
Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 8 hours run-time
GPU Notebook <= 8 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Submission file must be named submission.csv
Submission runtimes have been slightly obfuscated. If you repeat the exact same submission you will see up to 5 minutes of variance in the time before you receive your score.
Future Data Evaluation Phase
The run-time limits for both CPU and GPU notebooks will be extended to during the future data evaluation period proportional to the number of future samples. You must ensure your submission completes within that time. The extra runtime will enable us to use a substantially larger test set as the basis for ranking submissions on the final private leaderboard.

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

Additional Resources
What's the state-of-the-art in RNA 3D structure prediction? 2024 CASP16 challenge, including presentations from this competition's hosts Latest results from RNA-Puzzles, including predictions from this competition's hosts

The RibonanzaNet foundation model Ribonanza: deep learning of RNA structure through dual crowdsourcing

Stanford Ribonanza RNA Folding Kaggle challenge

How to think about RNA structure A perspective from domain experts

Cartoon explanation (XKCD) XKCD RNA

Citation
Shujun He, CASP16 organizers, CASP16 RNA experimentalists, RNA-Puzzles consortium, VFOLD team, Rachael Kretsch, Alissa Hummer, Andrew Favor, Walter Reade, Maggie Demkin, Rhiju Das, et al. Stanford RNA 3D Folding. https://kaggle.com/competitions/stanford-rna-3d-folding, 2025. Kaggle.


-----------------------------------

Dataset Description
In this competition you will predict five 3D structures for each RNA sequence.

Competition Phases and Updates
This is a code competition that will proceed in three phases.

Initial model training phase. At launch, there were approximately 25 sequences in the hidden test set. Some of those sequences were used for a private leaderboard to allow the host to track progress on wholly unseen data. During this phase the public test set sequences included–but was not limited to–targets from the 2024 CASP16 competition whose structures have not yet been publicly released in the PDB database.
Model training phase 2. On April 23, 2025, we updated the hidden test set and reset the leaderboard. Sequences in the current public test set were added to the train data, sequences currently in the private set were rolled into the new public set, and new sequences were added to the public test set.
Future data phase. Your selected submissions will be run against a completely new private test set generated after the end of the model training phases. There will be up to 40 sequences in the test set, all of them used for the private leaderboard.
Files
[train/validation/test]_sequences.csv - the target sequences of the RNA molecules.

target_id - (string) An arbitrary identifier. In train_sequences.csv, this is formatted as pdb_id_chain_id, where pdb_id is the id of the entry in the Protein Data Bank and chain_id is the chain id of the monomer in the pdb file.
sequence - (string) The RNA sequence. For test_sequences.csv, this is guaranteed to be a string of A, C, G, and U. For some train_sequences.csv, other characters may appear.
temporal_cutoff - (string) The date in yyyy-mm-dd format that the sequence was published. See Additional Notes.
description - (string) Details of the origins of the sequence. For a few targets, additional information on small molecule ligands bound to the RNA is included. You don't need to make predictions for these ligand coordinates.
all_sequences - (string) FASTA-formatted sequences of all molecular chains present in the experimentally solved structure. In a few cases this may include multiple copies of the target RNA (look for the word "Chains" in the header) and/or partners like other RNAs or proteins or DNA. You don't need to make predictions for all these molecules; if you do, just submit predictions for sequence. Some entries are blank.
[train/validation]_labels.csv - experimental structures.

ID - (string) that identifies the target_id and residue number, separated by _. Note: residue numbers use one-based indexing.
resname - (character) The RNA nucleotide ( A, C, G, or U) for the residue.
resid - (integer) residue number.
x_1,y_1,z_1,x_2,y_2,z_2,… - (float) Coordinates (in Angstroms) of the C1' atom for each experimental RNA structure. There is typically one structure for the RNA sequence, and train_labels.csv curates one structure for each training sequence. However, in some targets the experimental method has captured more than one conformation, and each will be used as a potential reference for scoring your predictions. validation_labels.csv has examples of targets with multiple reference structures (x_2,y_2,z_2, etc.).
train_[sequences/labels].v2.csv - extracted from the protein data bank with full text search for keyword RNA relaxed filter for unstructured RNAs based on pairwise C1' distances, where 20% of residues have to be close to some other residue that is over 4 bases apart.

sample_submission.csv

Same format as train_labels.csv but with five sets of coordinates for each of your five predicted structures (x_1,y_1,z_1,x_2,y_2,z_2,…x_5,y_5,z_5).
You must submit five sets of coordinates.
MSA/ and MSA_v2/ contain multiple sequence alignments in FASTA format for each target in train_sequences.csv and train_sequences.v2.csv. Files are named {target_id}.MSA.fasta. During evaluation with hidden test sequences, your notebook will have access to these MSA files for the test sequences.

PDB_RNA/ contains 3D structural information available in the Protein DataBase with

{PDB_id}.cif files for each RNA-containing entry
pdb_seqres_NA.fasta - sequences of all nucleic acid chains in the PDB in FASTA format.
pdb_release_dates_NA.csv - Entry ID and Release dates of the RNA-containing PDB entries in csv format. During the Future data phase, this folder will be updated to include all RNA structure entries in the PDB at the date of future evaluation, including entries released after the Final Submission Deadline.
Additional notes
The validation_sequences.csv and test_sequences.csv publicly provided here comprise 12 targets from the 2022 CASP15 competition which have been a widely used test set in the RNA modeling field.
If you choose to use the provided 12 CASP15 targets in validation_sequences.csv for validation, make sure that you train only on train_sequences.csv that have temporal_cutoff before the test_sequences (2022-05-27 is a safe date). If you wish, you can use train_sequences.csv with temporal_cutoff after this date as an additional validation set.
Once you begin hill climbing on the competition's actual Public Leaderboard, you can use all the train_sequences.csv and indeed all 3D structural information that you can find in the PDB database, since the competition's actual leaderboard targets are not released in the PDB database. However, note that the 12 CASP15 targets provided here in validation_sequences.csv will be 'burned' since they will be in your training set.
RNA chains from the same or different PDB entries that share sequence are given as different entries in train_sequences.csv. You may consider deduplicating these entries and merging the various available structures into additional x_2,y_2,z_2, etc. labels, as has been done with validation_sequences.csv
If you use RibonanzaNet (as in the competition starting notebook) it does not use information from the PDB before CASP15 and so is expected to be valid for use for all test sets. If you are using other neural networks, make sure to check their temporal cutoffs for training data.
If you are prompting a large language model you should request information that is available before the temporal_cutoff for each target. Otherwise, information from preprints or blog posts on CASP16 targets that were released after CASP16 competition end (2024-09-18) may leak into your submissions, and you will get a Public Leaderboard score that may be deceptively inflated compared to the CASP16 expert baseline or your eventual Private Leaderboard score. Only notebooks that beat the CASP16 expert baseline while also paying close attention to temporal_cutoff will be eligible for the Early Sharing prizes!
Additional files
The developers of RFdiffusion have made available a synthetic data set of over 400,000 RNA structures here.


_____________________

