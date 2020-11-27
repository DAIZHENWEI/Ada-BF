import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from Bloom_filter import BloomFilter
import os



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")
parser.add_argument('--model_path', action="store", dest="model_path", type=str, required=True,
                    help="path of the model")
parser.add_argument('--num_group_min', action="store", dest="min_group", type=int, required=True,
                    help="Minimum number of groups")
parser.add_argument('--num_group_max', action="store", dest="max_group", type=int, required=True,
                    help="Maximum number of groups")
parser.add_argument('--size_of_Ada_BF', action="store", dest="M_budget", type=int, required=True,
                    help="memory budget")
parser.add_argument('--c_min', action="store", dest="c_min", type=float, required=True,
                    help="minimum ratio of the keys")
parser.add_argument('--c_max', action="store", dest="c_max", type=float, required=True,
                    help="maximum ratio of the keys")



results = parser.parse_args()
DATA_PATH = results.data_path
num_group_min = results.min_group
num_group_max = results.max_group
model_size = os.path.getsize(results.model_path)
R_sum = results.M_budget - model_size * 8
c_min = results.c_min
c_max = results.c_max


# DATA_PATH = './URL_data.csv'
# num_group_min = 8
# num_group_max = 12
# R_sum = 200000
# c_min = 1.8
# c_max = 2.1


'''
Load the data and select training data
'''
data = pd.read_csv(DATA_PATH)
negative_sample = data.loc[(data['label']==-1)]
positive_sample = data.loc[(data['label']==1)]
train_negative = negative_sample.sample(frac = 0.3)



'''
Plot the distribution of scores
'''
plt.style.use('seaborn-deep')

x = data.loc[data['label']==1,'score']
y = data.loc[data['label']==-1,'score']
bins = np.linspace(0, 1, 25)

plt.hist([x, y], bins, log=True, label=['Keys', 'non-Keys'])
plt.legend(loc='upper right')
plt.savefig('./Score_Dist.png')
plt.show()



def R_size(count_key, count_nonkey, R0):
    R = [0]*len(count_key)
    R[0] = max(R0,1)
    for k in range(1, len(count_key)):
        R[k] = max(int(count_key[k] * (np.log(count_nonkey[0]/count_nonkey[k])/np.log(0.618) + R[0]/count_key[0])), 1)
    return R


def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    c_set = np.arange(c_min, c_max+10**(-6), 0.1)
    FP_opt = train_negative.shape[0]

    for num_group in range(num_group_min, num_group_max+1):
        for c in c_set:
            ### Determine the thresholds
            thresholds = np.zeros(num_group + 1)
            thresholds[0] = -0.1
            thresholds[-1] = 1.1
            num_negative = train_negative.shape[0]
            tau = sum(c ** np.arange(0, num_group, 1))
            num_piece = int(num_negative / tau)
            score = np.sort(np.array(list(train_negative['score'])))

            for i in range(1, num_group):
                if thresholds[-i] > 0:
                    score_1 = score[score < thresholds[-i]]
                    if int(num_piece * c ** (i - 1)) <= len(score_1):
                        thresholds[-(i + 1)] = score_1[-int(num_piece * c ** (i - 1))]
                    else:
                        thresholds[-(i + 1)] = 0
                else:
                    thresholds[-(i + 1)] = 1

            count_nonkey = np.zeros(num_group)
            for j in range(num_group):
                count_nonkey[j] = sum((score >= thresholds[j]) & (score < thresholds[j + 1]))

            num_group_1 = sum(count_nonkey > 0)
            count_nonkey = count_nonkey[count_nonkey > 0]
            thresholds = thresholds[-(num_group_1 + 1):]

            ### Count the keys of each group
            url = positive_sample['url']
            score = positive_sample['score']

            count_key = np.zeros(num_group_1)
            url_group = []
            bloom_filter = []
            for j in range(num_group_1):
                count_key[j] = sum((score >= thresholds[j]) & (score < thresholds[j + 1]))
                url_group.append(url[(score >= thresholds[j]) & (score < thresholds[j + 1])])

            ### Search the Bloom filters' size
            R = np.zeros(num_group_1 - 1)
            R[:] = 0.5 * R_sum
            non_empty_ix = min(np.where(count_key > 0)[0])
            if non_empty_ix > 0:
                R[0:non_empty_ix] = 0
            kk = 1
            while abs(sum(R) - R_sum) > 200:
                if (sum(R) > R_sum):
                    R[non_empty_ix] = R[non_empty_ix] - int((0.5 * R_sum) * (0.5) ** kk + 1)
                else:
                    R[non_empty_ix] = R[non_empty_ix] + int((0.5 * R_sum) * (0.5) ** kk + 1)
                R[non_empty_ix:] = R_size(count_key[non_empty_ix:-1], count_nonkey[non_empty_ix:-1], R[non_empty_ix])
                if int((0.5 * R_sum) * (0.5) ** kk + 1) == 1:
                    break
                kk += 1

            Bloom_Filters = []
            for j in range(int(num_group_1 - 1)):
                if j < non_empty_ix:
                    Bloom_Filters.append([0])
                else:
                    Bloom_Filters.append(BloomFilter(count_key[j], R[j]))
                    Bloom_Filters[j].insert(url_group[j])

            ### Test URLs
            ML_positive = train_negative.loc[(train_negative['score'] >= thresholds[-2]), 'url']
            url_negative = train_negative.loc[(train_negative['score'] < thresholds[-2]), 'url']
            score_negative = train_negative.loc[(train_negative['score'] < thresholds[-2]), 'score']

            test_result = np.zeros(len(url_negative))
            ss = 0
            for score_s, url_s in zip(score_negative, url_negative):
                ix = min(np.where(score_s < thresholds)[0]) - 1
                if ix >= non_empty_ix:
                    test_result[ss] = Bloom_Filters[ix].test(url_s)
                else:
                    test_result[ss] = 0
                ss += 1
            FP_items = sum(test_result) + len(ML_positive)
            print('False positive items: %d, Number of groups: %d, c = %f' %(FP_items, num_group, round(c, 2)))
            if FP_opt > FP_items:
                FP_opt = FP_items
                Bloom_Filters_opt = Bloom_Filters
                thresholds_opt = thresholds
                non_empty_ix_opt = non_empty_ix

    return Bloom_Filters_opt, thresholds_opt, non_empty_ix_opt



'''
Implement disjoint Ada-BF
'''
if __name__ == '__main__':
    '''Stage 1: Find the hyper-parameters'''
    Bloom_Filters_opt, thresholds_opt, non_empty_ix_opt = Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample)

    '''Stage 2: Run Ada-BF on all the samples'''
    ### Test URLs
    ML_positive = negative_sample.loc[(negative_sample['score'] >= thresholds_opt[-2]), 'url']
    url_negative = negative_sample.loc[(negative_sample['score'] < thresholds_opt[-2]), 'url']
    score_negative = negative_sample.loc[(negative_sample['score'] < thresholds_opt[-2]), 'score']
    test_result = np.zeros(len(url_negative))
    ss = 0
    for score_s, url_s in zip(score_negative, url_negative):
        ix = min(np.where(score_s < thresholds_opt)[0]) - 1
        if ix >= non_empty_ix_opt:
            test_result[ss] = Bloom_Filters_opt[ix].test(url_s)
        else:
            test_result[ss] = 0
        ss += 1
    FP_items = sum(test_result) + len(ML_positive)
    FPR = FP_items/len(url_negative)
    print('False positive items: {}; FPR: {}; Size of quries: {}'.format(FP_items, FPR, len(url_negative)))
