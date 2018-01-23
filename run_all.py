import os
import utils
import category_vector as cv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='zcm', help='yl, b8, zcm')
options = parser.parse_args()

host = options.host
dataset = [
    # 'MR',
    # 'TREC',
    'SST1',
    # 'SST2',
    # 'AG',
    # 'yelpf',
    # 'yelpp'
]
we = [
    # 'w2v',
    # 'randPARAMn',
    # 'scv',
    # 'w2v_scv',
    # 'w2v_randPARAMn',
    # 'chi2scv',
    # 'w2v_chi2scv',
    # 'rfscv',
    # 'w2v_rfscv',
    # 'iqfscv',
    # 'w2v_iqfscv',
    # 'scvPARAM5000',
    # 'w2v_scvPARAM5000',
    # 'chi2scvPARAM5000',
    # 'w2v_chi2scvPARAM5000',
    # 'rfscvPARAM5000',
    # 'w2v_rfscvPARAM5000',
    # 'iqfscvPARAM5000',
    # 'w2v_iqfscvPARAM5000',
    # 'yelpfscv',
    # 'w2v_yelpfscv',
    # 'yelpfscvPARAM5000',
    # 'w2v_yelpfscvPARAM5000',
    # 'yelpfscv_scv',
    # 'yelpfscvPARAM5000_scvPARAM5000',
    # 'kcv',
    # 'w2v_kcv',
    # 'kcvPARAM1',
    # 'w2v_kcvPARAM1',
    # 'w2vkcv',
    # 'kcv_w2vkcv',
    # 'w2v_kcv_w2vkcv',
    # 'w2v_scv_kcv_w2vkcv',
    # 'w2v_scvPARAM5000_kcv_w2vkcv',
    # 'scvPARAM5000_kcv_w2vkcv',
    # 'scvPARAM5000_kcv',
    # 'scvPARAM5000_w2vkcv',
    # 'kcvPARAM1_w2vkcv',
    # 'w2v_kcvPARAM1_w2vkcv',
    # 'w2v_scv_kcvPARAM1_w2vkcv',
    # 'w2v_scvPARAM5000_kcvPARAM1_w2vkcv',
    # 'scvPARAM5000_kcvPARAM1_w2vkcv',
    # 'scvPARAM5000_kcvPARAM1',
    # 'senticv',
    'w2v_senticv',
    # 'senticvPARAM5000',
    # 'w2v_senticvPARAM5000',
    # 'w2v_senticv_scv',
    # 'w2v_senticvPARAM5000_scvPARAM5000'
]
filters = [
    ['4', '5', '6'],
    ['3', '4', '5'],
    ['3', '4', '5', '6'],
    ['2', '3', '4']
    ]
filter_num = [
    ['50', '50', '50'],
    ['100', '100', '100'],
    ['75', '75', '75']
    ]
dropout_prob = [
    0.5,
    0.6,
    0.4,
    0.3,
    0.2,
    ]
norm_limit = [
    1,
    2,
    3,
    4,
    5,
    # 0.00001
    ]
batch_size = [
    # 5,
    # 10,
    # 20,
    # 30,
    # 40,
    50,
    # 60,
    # 100
    ]
gpu = {'MR': -0, 'TREC': -0, 'SST1': -0, 'SST2': 0, 'AG': 0}
result_path = {d: 'result/result_{}_clean_{}.csv'.format(d, host) for d in dataset}
we = [x.replace('_', '-') for x in we]

for d in dataset:
    for w in we:
        for f in filters:
            for fn in filter_num:
                for dp in dropout_prob:
                    for nl in norm_limit:
                        for bs in batch_size:
                            print(d, w, f, fn, dp, nl, bs)
                            os.system('python run.py --type CNN3 --dataset {} --early_stopping --we {} --filters {} --filter_num {} --dropout_prob {} --norm_limit {} --batch_size {} --gpu {} --result_path {}'
                                      .format(d, w, ' '.join(f), ' '.join(fn), dp, nl, bs, gpu[d], result_path[d]))
    # data = utils.get_dataset(d)
    # cv.load_w2v(data)
