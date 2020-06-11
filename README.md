# ADAPTIVE LEARNED BLOOM FILTER (ADA-BF): EFFICIENT UTILIZATION OF THE CLASSIFIER

The python files include the implementation of the Bloom filter, learned Bloom filter, Ada-BF and disjoint Ada-BF, and print the size of False Positives of the corresponding algorithm.

**Input arguments**: 
- `--data_path`: a csv file includes the items, scores and labels; `--size_of_Ada_BF`: size of Bloom filter;
- (for learned Bloom filter) `--threshold_min` and `--threshold_max` provide the range of the score threshold (between `threshold_min` and `threshold_max`). Items with score larger than the threshold are identified as keys;
- (for Ada-BF and disjoint Ada-BF) `--num_group_min` and `--num_group_max` give the range of number of groups to divide (range of *g*
); `--c_min` and `--c_max` provide the range of *c* where *c=m_j/m_{j+1}*

**Commands**:
- Run Bloom filter: `python Bloom_filter.py --data_path ./Datasets/URL_data.csv --size_of_Ada_BF 200000`
- Run learned Bloom filter: `python learned_Bloom_filter.py --data_path ./Datasets/URL_data.csv --size_of_Ada_BF 200000  --threshold_min 0.5   --threshold_max 0.95`
