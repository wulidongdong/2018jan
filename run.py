import model as m
import utils

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import numpy as np
import argparse
import copy
from operator import itemgetter
import category_vector as cv
import pandas as pd
import time
import os
# from imp import reload
# reload(utils)


def early_stopping(dev_acc, dev_acc_list):
    if len(dev_acc_list) > 10 and sum([(dev_acc - x) <= 0 for x in dev_acc_list[-5:]]) > 3:
        dev_acc_list.append(dev_acc)
        return True
    dev_acc_list.append(dev_acc)
    return False


def train(data, params):
    # params = {'we': 'scv', 'WORD_DIM': 300}
    models = []
    parameters = []
    optimizer = []
    criterion = []
    for modeli, t in enumerate(params['type'].split('_')):
        if t == 'CNN2':
            wv_matrix = [1, 1]
            wv_matrix[0] = cv.load_w2v(data)
            wv_matrix[1] = cv.load_scv(data)
            params['WV_MATRIX'] = wv_matrix
            params['WORD_DIM'] = [x.shape[1] for x in wv_matrix]
            params['NEW_WORD_DIM'] = 300

        elif t == 'CNN3':
            wv_matrix = [1, 1]
            for index, we in enumerate(params['we'].split('-')):
                if 'PARAM' in we:
                    p = we.split('PARAM')[1]
                    we = we.split('PARAM')[0]
                    wv_matrix[index] = getattr(cv, 'load_{}'.format(we))(data, p)
                else:
                    wv_matrix[index] = getattr(cv, 'load_{}'.format(we))(data)
            params['WV_MATRIX'] = wv_matrix
            params['WORD_DIM'] = [x.shape[1] for x in wv_matrix]
        else:
            for index, we in enumerate(params['we'].split('-')[modeli].split('_')):
                print('load {} for {} - {}'.format(we, modeli, t))
                if 'PARAM' in we:
                    p = we.split('PARAM')[1]
                    we = we.split('PARAM')[0]
                    one_matrix = getattr(cv, 'load_{}'.format(we))(data, p)
                else:
                    one_matrix = getattr(cv, 'load_{}'.format(we))(data)
                if index == 0:
                    wv_matrix = one_matrix
                else:
                    wv_matrix = np.concatenate([wv_matrix, one_matrix], axis=1)
            params['WV_MATRIX'] = wv_matrix
            params['WORD_DIM'] = wv_matrix.shape[1]
            params['idx_to_word'] = data['idx_to_word']
            params['word_to_idx'] = data['word_to_idx']
            if t == 'CNNCCA':
                params['concept_vectors'] = data['concept_vectors']

        if params['GPU'] == -1:
            print('using cpu')
            models.append(getattr(m, t)(**params))
        else:
            print('using gpu')
            models.append(getattr(m, t)(**params).cuda(params['GPU']))

        parameters.append(filter(lambda p: p.requires_grad, models[modeli].parameters()))
        # weight_decay = params['NORM_LIMIT'] if isinstance(models[modeli], m.BLSTM) else 0
        weight_decay = 0
        print('weight_decay', weight_decay)
        optimizer.append(optim.Adadelta(parameters[modeli], params['LEARNING_RATE'], weight_decay=weight_decay))
        criterion.append(nn.CrossEntropyLoss())

    # pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0
    best_test_echo = 0
    dev_acc_list = []
    for e in range(params['EPOCH']):
        data['train_x'], data['train_y'] = shuffle(data['train_x'], data['train_y'])

        for i in range(0, len(data['train_x']), params['BATCH_SIZE']):
            batch_range = min(params['BATCH_SIZE'], len(data['train_x']) - i)

            batch_x = [[data['word_to_idx'][w] for w in sent] +
                       [params['VOCAB_SIZE'] + 1] * (params['MAX_SENT_LEN'] - len(sent))
                       for sent in data['train_x'][i:i + batch_range]]
            batch_y = [data['classes'].index(c) for c in data['train_y'][i:i + batch_range]]

            if params['GPU'] == -1:
                batch_x = Variable(torch.LongTensor(batch_x))
                batch_y = Variable(torch.LongTensor(batch_y))
            else:
                batch_x = Variable(torch.LongTensor(batch_x)).cuda(params['GPU'])
                batch_y = Variable(torch.LongTensor(batch_y)).cuda(params['GPU'])
            for modeli, model in enumerate(models):
                optimizer[modeli].zero_grad()
                model.train()
                pred = model(batch_x)
                loss = criterion[modeli](pred, batch_y)
                loss.backward()
                if params['NORM_LIMIT'] > 0:
                    nn.utils.clip_grad_norm(parameters[modeli], max_norm=params['NORM_LIMIT'])
                optimizer[modeli].step()

        if any([model.name in ['BLSTM', 'AttBLSTM'] for model in models]):
            print('batch_test')
            dev_acc = batch_test(data, models, params, mode='dev')
            test_acc = batch_test(data, models, params)
        else:
            dev_acc = test(data, models, params, mode='dev')
            test_acc = test(data, models, params)
        print('epoch:', e + 1, '/ dev_acc:', dev_acc, '/ test_acc:', test_acc)
        # print(model.embedding2.weight.data.numpy())
        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            best_model = copy.deepcopy(model)

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            best_test_echo = e

        if params['EARLY_STOPPING'] and early_stopping(dev_acc, dev_acc_list):
            print('early stopping by dev_acc!')

            break
        # else:
        #     pre_dev_acc = dev_acc

    print('max dev acc:', max_dev_acc, 'test acc:', max_test_acc)
    columns = ['datetime', 'MODEL', 'DATASET', 'VOCAB_SIZE', 'STOP_EPOCH', 'BEST_ECHO', 'LEARNING_RATE', 'EARLY_STOPPING', 'SAVE_MODEL', 'WORD_EMBEDDING', 'MODEL_TYPE',
               'BATCH_SIZE', 'FILTERS', 'FILTER_NUM', 'DROPOUT_PROB', 'NORM_LIMIT', 'max_dev_acc', 'max_test_acc']
    if os.path.exists(params['result_path']):
        result = pd.read_csv(params['result_path'])
    else:
        result = pd.DataFrame(columns=columns)
    result = result.append(pd.DataFrame([(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())),
                                          params['MODEL'],
                                          params['DATASET'],
                                          params['VOCAB_SIZE'],
                                          e,
                                          best_test_echo,
                                          params['LEARNING_RATE'],
                                          params['EARLY_STOPPING'],
                                          params['SAVE_MODEL'],
                                          params['we'],
                                          params['type'],
                                          params['BATCH_SIZE'],
                                          params['FILTERS'],
                                          params['FILTER_NUM'],
                                          params['DROPOUT_PROB'],
                                          params['NORM_LIMIT'],
                                          max_dev_acc,
                                          max_test_acc)], columns=columns))
    result.to_csv(params['result_path'], index=False)
    return best_model


def test(data, models, params, mode='test'):

    if mode == 'dev':
        x, y = data['dev_x'], data['dev_y']
    elif mode == 'test':
        x, y = data['test_x'], data['test_y']

    x = [[data['word_to_idx'][w] if w in data['vocab'] else params['VOCAB_SIZE'] for w in sent] +
         [params['VOCAB_SIZE'] + 1] * (params['MAX_SENT_LEN'] - len(sent))
         for sent in x]

    if params['GPU'] == -1:
        x = Variable(torch.LongTensor(x))
    else:
        x = Variable(torch.LongTensor(x)).cuda(params['GPU'])
    y = [data['classes'].index(c) for c in y]
    pred = []
    for modeli, model in enumerate(models):
        model.eval()
        pred.append(model(x).cpu().data.numpy())

    pred = np.array(pred).reshape(len(models), -1)
    pred = np.mean(pred, 0).reshape(-1, len(data['classes']))
    pred = np.argmax(pred, axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc


def batch_test(data, models, params, mode='test'):

    if mode == 'dev':
        x, y = data['dev_x'], data['dev_y']
    elif mode == 'test':
        x, y = data['test_x'], data['test_y']

    acc = 0
    for i in range(0, len(x), params['BATCH_SIZE']):
        batch_range = min(params['BATCH_SIZE'], len(x) - i)

        batch_x = [[data['word_to_idx'][w] for w in sent] +
                   [params['VOCAB_SIZE'] + 1] * (params['MAX_SENT_LEN'] - len(sent))
                   for sent in x[i:i + batch_range]]
        batch_y = [data['classes'].index(c) for c in y[i:i + batch_range]]

        if params['GPU'] == -1:
            batch_x = Variable(torch.LongTensor(batch_x))
        else:
            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params['GPU'])
            pred = models[0](batch_x).cpu().data.numpy()
            assert(len(pred) == len(batch_y))
            pred = np.argmax(pred, axis=1)
            acc += sum([1 if p == by else 0 for p, by in zip(pred, batch_y)])

    return acc / len(y)


def main():
    parser = argparse.ArgumentParser(description='-----[CNN-classifier]-----')
    parser.add_argument('--mode', default='train', help='train: train (with test) a model / test: test saved models')
    parser.add_argument('--model', default='non-static', help='available models: rand, static, non-static, multichannel')
    parser.add_argument('--dataset', default='TREC', help='available datasets: MR, TREC, AG, SST1, SST2')
    parser.add_argument('--save_model', default=False, action='store_true', help='whether saving model or not')
    parser.add_argument('--early_stopping', default=False, action='store_true', help='whether to apply early stopping')
    parser.add_argument('--epoch', default=100, type=int, help='number of max epoch')
    parser.add_argument('--learning_rate', default=1.0, type=float, help='learning rate')
    parser.add_argument('--gpu', default=-1, type=int, help='the number of gpu to be used')
    parser.add_argument('--cv', default=False, action='store_true', help='whether to use cross validation')
    parser.add_argument('--we', default='w2v', help='available word embedding: w2v, rand, scv, yelpfscv')
    parser.add_argument('--type', default='CNN', help='available type for cnn model: CNN, CNN2')
    parser.add_argument('--batch_size', default=50, type=int, help='batch_size')
    parser.add_argument('--filters', default=[3, 4, 5], type=int, nargs='+', help='filters')
    parser.add_argument('--filter_num', default=[100, 100, 100], type=int, nargs='+', help='filter_num')
    parser.add_argument('--dropout_prob', default=0.5, type=float, help='dropout_prob')
    parser.add_argument('--norm_limit', default=3, type=float, help='norm_limit')
    parser.add_argument('--result_path', default='result/result_auto.csv', help='result_path')

    options = parser.parse_args()

    data = utils.get_dataset(options.dataset)
    # data = utils.get_dataset('MR')
    params = {
        'MODEL': options.model,
        'DATASET': options.dataset,
        'SAVE_MODEL': options.save_model,
        'EARLY_STOPPING': options.early_stopping,
        'EPOCH': options.epoch,
        'LEARNING_RATE': options.learning_rate,
        'MAX_SENT_LEN': max([len(sent) for sent in data['train_x'] + data['dev_x'] + data['test_x']]),
        'BATCH_SIZE': options.batch_size,
        'WORD_DIM': 300,
        'VOCAB_SIZE': len(data['vocab']),
        'CLASS_SIZE': len(data['classes']),
        'FILTERS': options.filters,
        'FILTER_NUM': options.filter_num,
        'DROPOUT_PROB': options.dropout_prob,
        'NORM_LIMIT': options.norm_limit,
        'GPU': options.gpu,
        'cv': options.cv,
        'we': options.we,
        'type': options.type,
        'result_path': options.result_path
    }

    print('=' * 20 + 'INFORMATION' + '=' * 20)
    print('MODEL:', params['MODEL'])
    print('DATASET:', params['DATASET'])
    print('VOCAB_SIZE:', params['VOCAB_SIZE'])
    print('EPOCH:', params['EPOCH'])
    print('LEARNING_RATE:', params['LEARNING_RATE'])
    print('EARLY_STOPPING:', params['EARLY_STOPPING'])
    print('SAVE_MODEL:', params['SAVE_MODEL'])
    print('WORD EMBEDDING:', params['we'])
    print('MODEL TYPE:', params['type'])
    print('BATCH_SIZE', params['BATCH_SIZE'])
    print('FILTERS', params['FILTERS'])
    print('FILTER_NUM', params['FILTER_NUM'])
    print('DROPOUT_PROB', params['DROPOUT_PROB'])
    print('NORM_LIMIT', params['NORM_LIMIT'])
    print('=' * 20 + 'INFORMATION' + '=' * 20)

    if options.mode == 'train':
        print('=' * 20 + 'TRAINING STARTED' + '=' * 20)
        if params['cv']:
            kf = KFold(n_splits=10, shuffle=True)
            for index, (train_index, test_index) in enumerate(kf.split(data['x'])):
                print('cv {}'.format(index))
                train_index = shuffle(train_index)
                dev_size = len(train_index) // 10
                dev_index = train_index[:dev_size]
                train_index = train_index[dev_size:]
                data['dev_x'] = list(itemgetter(*dev_index)(data['x']))
                data['train_x'] = list(itemgetter(*train_index)(data['x']))
                data['test_x'] = list(itemgetter(*test_index)(data['x']))
                data['dev_y'] = list(itemgetter(*dev_index)(data['y']))
                data['train_y'] = list(itemgetter(*train_index)(data['y']))
                data['test_y'] = list(itemgetter(*test_index)(data['y']))
                train(data, params)
        else:
            model = train(data, params)
            if params['SAVE_MODEL']:
                utils.save_model(model, params)
        print('=' * 20 + 'TRAINING FINISHED' + '=' * 20)
    else:
        if params['GPU'] == -1:
            model = utils.load_model(params)
        else:
            model = utils.load_model(params).cuda(params['GPU'])

        test_acc = test(data, model, params)
        print('test acc:', test_acc)


if __name__ == '__main__':
    main()
