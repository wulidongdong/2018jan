from sklearn.utils import shuffle

import pickle
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_dataset(name):
    data = eval('read_{}'.format(name))()
    if data['lemmatize']:
        lemmatizer = WordNetLemmatizer()
        data['train_x'] = [[lemmatizer.lemmatize(x) for x in y] for y in data['train_x']]
        data['dev_x'] = [[lemmatizer.lemmatize(x) for x in y] for y in data['dev_x']]
        data['test_x'] = [[lemmatizer.lemmatize(x) for x in y] for y in data['test_x']]
        print('lemmatizer')
    data['vocab'] = sorted(list(set([w for sent in data['train_x'] + data['dev_x'] + data['test_x'] for w in sent])))
    data['classes'] = sorted(list(set(data['train_y'])))
    data['word_to_idx'] = {w: i for i, w in enumerate(data['vocab'])}
    data['idx_to_word'] = {i: w for i, w in enumerate(data['vocab'])}
    data['name'] = name
    return data


def read_yelpf():
    data = {}

    def read(mode):
        x = pd.read_csv('data/yelp_f/' + mode + '.csv')
        y = x.iloc[:, 0].values.tolist()
        x = x.iloc[:, 1].apply(lambda x: clean_str(x).split()).values.tolist()

        x, y = shuffle(x, y)

        if mode == 'train':
            dev_idx = len(x) // 10
            data['dev_x'], data['dev_y'] = x[:dev_idx], y[:dev_idx]
            data['train_x'], data['train_y'] = x[dev_idx:], y[dev_idx:]
        else:
            data['test_x'], data['test_y'] = x, y

    read('train')
    read('test')
    data['lemmatize'] = True
    return data


def read_yelpp():
    data = {}

    def read(mode):
        x = pd.read_csv('data/yelp_p/' + mode + '.csv')
        y = x.iloc[:, 0].values.tolist()
        x = x.iloc[:, 1].apply(lambda x: clean_str(x).split()).values.tolist()

        x, y = shuffle(x, y)

        if mode == 'train':
            dev_idx = len(x) // 10
            data['dev_x'], data['dev_y'] = x[:dev_idx], y[:dev_idx]
            data['train_x'], data['train_y'] = x[dev_idx:], y[dev_idx:]
        else:
            data['test_x'], data['test_y'] = x, y

    read('train')
    read('test')
    data['lemmatize'] = True
    return data


def read_SST1():
    data = {}

    def read(mode):
        x, y = [], []

        with open('data/SST-1/sst5_' + mode + '.csv', 'r', encoding='utf-8') as f:
            for line in f:
                if line[-1] == '\n':
                    line = line[:-1]
                y.append(line.split()[-1])
                x.append(clean_str(line).split()[:-1])

        x, y = shuffle(x, y)

        data[mode + '_x'] = x
        data[mode + '_y'] = y

    read('train')
    read('test')
    read('dev')
    data['lemmatize'] = True
    return data


def read_SST2():
    data = {}

    def read(mode):
        x, y = [], []

        with open('data/SST-2/sst2_' + mode + '.csv', 'r', encoding='utf-8') as f:
            for line in f:
                if line[-1] == '\n':
                    line = line[:-1]
                y.append(line.split()[-1])
                x.append(clean_str(line).split()[:-1])

        x, y = shuffle(x, y)

        data[mode + '_x'] = x
        data[mode + '_y'] = y

    read('train')
    read('test')
    read('dev')
    data['lemmatize'] = True
    return data


def read_AG():
    data = {}

    def read(mode):
        x, y = [], []

        with open('data/AG/' + mode + '.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if line[-1] == '\n':
                    line = line[:-1]
                y.append(line.split()[0])
                x.append(clean_str(line).split()[1:])

        x, y = shuffle(x, y)

        if mode == 'train':
            dev_idx = len(x) // 10
            data['dev_x'], data['dev_y'] = x[:dev_idx], y[:dev_idx]
            data['train_x'], data['train_y'] = x[dev_idx:], y[dev_idx:]
        else:
            data['test_x'], data['test_y'] = x, y

    data['labels'] = ['world', 'business', 'sport', 'science', 'technology']
    read('train')
    read('test')
    data['lemmatize'] = True
    return data


def read_TREC():
    data = {}

    def read(mode):
        x, y = [], []
        with open('data/TREC/TREC_' + mode + '.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if line[-1] == '\n':
                    line = line[:-1]
                y.append(line.split()[0].split(':')[0])
                # line = ' '.join(line.split()[1:])
                x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == 'train':
            dev_idx = len(x) // 10
            data['dev_x'], data['dev_y'] = x[:dev_idx], y[:dev_idx]
            data['train_x'], data['train_y'] = x[dev_idx:], y[dev_idx:]
        else:
            data['test_x'], data['test_y'] = x, y

    read('train')
    read('test')
    data['lemmatize'] = True
    return data


def read_MR():
    data = {}
    x, y = [], []

    with open('data/MR/rt-polarity.pos', 'r', encoding='utf-8') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            x.append(clean_str(line).split())
            y.append(1)

    with open('data/MR/rt-polarity.neg', 'r', encoding='utf-8') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            x.append(clean_str(line).split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data['train_x'], data['train_y'] = x[:dev_idx], y[:dev_idx]
    data['dev_x'], data['dev_y'] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data['test_x'], data['test_y'] = x[test_idx:], y[test_idx:]
    data['x'] = x
    data['y'] = y
    data['lemmatize'] = True
    return data


def save_model(model, params):
    path = 'saved_models/{}_{}_{}.pkl'.format(params['DATASET'], params['MODEL'], params['EPOCH'])
    pickle.dump(model, open(path, 'wb'))
    print('A model is saved successfully as {}!'.format(path))


def load_model(params):
    path = 'saved_models/{}_{}_{}.pkl'.format(params['DATASET'], params['MODEL'], params['EPOCH'])

    try:
        model = pickle.load(open(path, 'rb'))
        print('Model in {} loaded successfully!'.format(path))

        return model
    except:
        print('No available model such as {}.'.format(path))
        exit()
