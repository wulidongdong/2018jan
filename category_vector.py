import numpy as np
from collections import Counter
import itertools
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet_ic
from tqdm import tqdm
import utils
import os
import pickle
import scipy.spatial as sp
import json
import nltk


def normalize(matrix):
    matrix = np.array(matrix)
    matrix = matrix / matrix.sum(1)[:, None]
    matrix[np.isnan(matrix)] = 0
    # ss = StandardScaler()
    # matrix_std = ss.fit_transform(matrix.T)
    return matrix


def load_rand(data, dim=300):
    if dim == 'n':
        dim = len(data['classes'])
    else:
        dim = int(dim)
    rand = []
    for i in range(len(data['vocab'])):
        rand.append(np.random.uniform(-0.01, 0.01, dim).astype('float32'))

    rand.append(np.random.uniform(-0.01, 0.01, dim).astype('float32'))
    rand.append(np.zeros(dim).astype('float32'))
    rand = np.array(rand)
    return rand


def get_max_feature(vectors, data, max_feature, new_value, dim=2):
    counter_sort = get_max_feature_list(data, max_feature)
    assert(len(vectors) == len(data['vocab']))
    for i in range(len(data['vocab'])):
        word = data['idx_to_word'][i]
        if word not in counter_sort:
            vectors[i] = eval(new_value)
    return vectors


def get_max_feature_list(data, max_feature):
    data_x = np.array(data['train_x'])
    if max_feature == 'prob':
        counter_sort = get_prob(data_x)
    else:
        counter_all = Counter(list(itertools.chain.from_iterable(data_x)))
        counter_sort = [x[0] for x in sorted(counter_all.items(), key=lambda x: x[1], reverse=True)[:int(max_feature)]]
    return counter_sort


def softmax(x):
    x = np.array(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_prob(data_x):
    counter_all = Counter(list(itertools.chain.from_iterable(data_x)))
    counter_sort = sorted(counter_all.items(), key=lambda x: x[1], reverse=True)
    counter_len = len(counter_sort)
    print('counter_len', counter_len)
    counter_prob = [x[1] for x in counter_sort]
    denominator = counter_prob[0]
    counter_prob = [x / denominator for x in counter_prob]
    # counter_prob = softmax(counter_prob)
    counter_sort = [(x[0], y) for x, y in zip(counter_sort, counter_prob)]
    print(counter_sort[:100])
    counter_sort = [x[0] if np.random.rand() < x[1] else 0 for x in counter_sort]
    counter_sort = list(filter(lambda x: x != 0, counter_sort))
    print('counter_sort', len(counter_sort))
    return counter_sort


def get_doc_category(data_category):
    doc_category_one = Counter(list(itertools.chain.from_iterable(data_category)))
    for k in doc_category_one.keys():
        count = 0
        for x in data_category:
            if k in x:
                count += 1
        doc_category_one[k] = count

    return doc_category_one, len(data_category)


def get_category_value(data, method='tf'):

    data_x = np.array(data['train_x'])
    data_y = np.array(data['train_y'])

    if method == 'tf':
        print('using tf for scv')
        tf_category = {}
        for c in data['classes']:
            tf_category[c] = Counter(list(itertools.chain.from_iterable(data_x[np.where(data_y == c)])))
        return tf_category
    elif method == 'chi2':
        print('using chi2 for scv')
        doc_category = {}
        len_category = {}
        tp = {}
        fp = {}
        fn = {}
        tn = {}
        chi2_category = {}
        for c in data['classes']:
            doc_category[c], len_category[c] = get_doc_category(data_x[np.where(data_y == c)])
        for c in data['classes']:
            tp[c] = doc_category[c].copy()
            fp[c] = {k: (len_category[c] - v) for k, v in doc_category[c].items()}
            fn[c] = {k: sum([doc_category[dk][k] if dk != c else 0 for dk, dv in doc_category.items()]) for k, v in doc_category[c].items()}
            tn[c] = {k: sum([len_category[dk] - doc_category[dk][k] if dk != c else 0 for dk, dv in doc_category.items()]) for k, v in doc_category[c].items()}
            chi2_category[c] = {k: sum(len_category.values()) * pow((tp[c][k] * tn[c][k] - fp[c][k] * fn[c][k]), 2) /
                                ((tp[c][k] + fp[c][k]) * (fn[c][k] + tn[c][k]) * (tp[c][k] + fn[c][k]) * (fp[c][k] + tn[c][k])) for k in doc_category[c].keys()}
        return chi2_category

    elif method == 'rf':
        print('using rf for scv')
        doc_category = {}
        len_category = {}
        tp = {}
        fn = {}
        rf_category = {}
        tf_category = {}
        for c in data['classes']:
            doc_category[c], len_category[c] = get_doc_category(data_x[np.where(data_y == c)])
            tf_category[c] = Counter(list(itertools.chain.from_iterable(data_x[np.where(data_y == c)])))
        for c in data['classes']:
            tp[c] = doc_category[c].copy()
            fn[c] = {k: sum([doc_category[dk][k] if dk != c else 0 for dk, dv in doc_category.items()]) for k, v in doc_category[c].items()}
            rf_category[c] = {k: tf_category[c][k] * np.log(2 + ((tp[c][k] + 1) / (fn[c][k] + 1))) for k in doc_category[c].keys()}
        return rf_category
    elif method == 'iqf':
        print('using iqf for scv')
        doc_category = {}
        len_category = {}
        tp = {}
        fn = {}
        iqf_category = {}
        cf = Counter(list(itertools.chain.from_iterable(data_x)))
        for c in data['classes']:
            doc_category[c], len_category[c] = get_doc_category(data_x[np.where(data_y == c)])
        for k, v in cf.items():
            cf[k] = sum([1 if k in doc_category[d].keys() else 0 for d in doc_category.keys()])
        for c in data['classes']:
            tp[c] = doc_category[c].copy()
            fn[c] = {k: sum([doc_category[dk][k] if dk != c else 0 for dk, dv in doc_category.items()]) for k, v in doc_category[c].items()}
            iqf_category[c] = {k: np.log(sum(len_category.values()) / (tp[c][k] + fn[c][k]))
                               * np.log(tp[c][k] + 1)
                               * np.log(len(data['classes']) / cf[k] + 1)
                               for k in doc_category[c].keys()}
        return iqf_category


def load_tfscv(data, max_feature=None):
    return load_scv(data, max_feature, method='tf')


def load_chi2scv(data, max_feature=None):
    return load_scv(data, max_feature, method='chi2')


def load_rfscv(data, max_feature=None):
    return load_scv(data, max_feature, method='rf')


def load_iqfscv(data, max_feature=None):
    return load_scv(data, max_feature, 'iqf')


def load_scv(data, max_feature=None, method='tf', normal=True):
    category = get_category_value(data, method=method)

    mean_tf = sum(Counter(list(itertools.chain.from_iterable(data['train_x']))).values()) // len(data['train_x'])
    print('mean_tf', mean_tf)
    new_value_list = [mean_tf] * len(data['classes'])
    scv = []
    for i in range(len(data['vocab'])):
        word = data['idx_to_word'][i]
        word_vector = [category[c][word] if word in category[c].keys() else 0 for c in data['classes']]
        word_sum = sum(word_vector)
        if word_sum != 0:
            scv.append(word_vector)
        else:
            scv.append(new_value_list)
    if max_feature is not None:
        scv = get_max_feature(scv, data, max_feature, str(new_value_list))
    scv.append(new_value_list)
    scv.append(np.zeros(len(data['classes'])).astype('float32'))
    scv = np.array(scv)
    print(scv.shape)
    if normal:
        return normalize(scv)
    return scv


def get_concept_embed(data):
    path = 'temp/{}concept_embed.pkl'.format(data['name'])
    if os.path.exists(path):
        with open(path, 'rb') as f:
            concept_embed = pickle.load(f)
    else:
        with open('data/concept.json', 'r') as f:
            concept = json.load(f)
        word_vectors = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
        concept_embed = []
        for word in tqdm(data['word_to_idx'].keys()):
            if word in concept.keys():
                word_concepts = concept[word]
                for index, wc in enumerate(word_concepts):
                    if wc in word_vectors.vocab:
                        word_concepts[index] = word_vectors.word_vec(wc)
                    else:
                        word_concepts[index] = np.random.uniform(-0.01, 0.01, 300).astype('float32')
            else:
                word_concepts = [np.random.uniform(-0.01, 0.01, 300).astype('float32')]
            concept_embed.append(word_concepts)

        concept_embed.append([np.random.uniform(-0.01, 0.01, 300).astype('float32')])
        concept_embed.append([np.zeros(300).astype('float32')])
        concept_embed = np.array(concept_embed)
        print(len(concept_embed[0]))
        print(concept_embed[-1])
        with open(path, 'wb') as f:
            pickle.dump(concept_embed, f)
    return concept_embed


def load_w2vconcept(data):
    data['concept_vectors'] = {}
    wv_matrix = load_w2v(data)
    data['concept_vectors']['word_vectors'] = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
    with open('data/concept.json', 'r') as f:
        data['concept_vectors']['concept'] = json.load(f)
        print('concept vectors loaded')
    return wv_matrix


def load_w2v(data, exists=False):
    # load word2vec
    path = 'temp/' + data['name'] + 'w2v.pkl'
    list_path = 'temp/' + data['name'] + 'w2v_list.pkl'
    if os.path.exists(path):
        print('use cache {} w2v'.format(data['name']))
        with open(path, 'rb') as f:
            wv_matrix = pickle.load(f)
        with open(list_path, 'rb') as f:
            exists_list = pickle.load(f)
    else:
        exists_list = []
        # word_vectors = KeyedVectors.load_word2vec_format('/home/cike/deeplearning/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        word_vectors = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
        # word_vectors = KeyedVectors.load_word2vec_format('E:/deeplearning/w2v/GoogleNews-vectors-negative300.bin', binary=True)

        wv_matrix = []
        count = 0
        for i in range(len(data['vocab'])):
            word = data['idx_to_word'][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
                exists_list.append(1)
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype('float32'))
                count += 1
                exists_list.append(0)

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype('float32'))
        wv_matrix.append(np.zeros(300).astype('float32'))
        wv_matrix = np.array(wv_matrix)
        with open(path, 'wb') as f:
            pickle.dump(wv_matrix, f)
        with open(list_path, 'wb') as f:
            pickle.dump(exists_list, f)

        print(data['name'], len(data['vocab']), count)
    if exists:
        return wv_matrix, exists_list
    return wv_matrix


def load_glove(data, exists=False):
    path = 'temp/{}glove.pkl'.format(data['name'])
    list_path = 'temp/{}glove_list.pkl'.format(data['name'])
    if os.path.exists(path):
        print('use cache {} glove'.format(data['name']))
        with open(path, 'rb') as f:
            wv_matrix = pickle.load(f)
        with open(list_path, 'rb') as f:
            exists_list = pickle.load(f)
    else:
        exists_list = []
        # if not os.path.exists('data/glove_w2v.txt'):
        #     glove2word2vec(glove_input_file='data/glove.6B.300d.txt', word2vec_output_file='data/glove_w2v.txt')
        # word_vectors = KeyedVectors.load_word2vec_format('data/glove_w2v.txt', binary=False)
        with open('data/glove.6B.300d.txt', 'r') as f:
            word_vectors = {}
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                word_vectors[word] = embedding
        wv_matrix = []
        count = 0
        for i in range(len(data['vocab'])):
            word = data['idx_to_word'][i]
            if word in word_vectors.keys():
                wv_matrix.append(word_vectors[word])
                exists_list.append(1)
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype('float32'))
                count += 1
                exists_list.append(0)

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype('float32'))
        wv_matrix.append(np.zeros(300).astype('float32'))
        wv_matrix = np.array(wv_matrix)
        with open(path, 'wb') as f:
            pickle.dump(wv_matrix, f)
        with open(list_path, 'wb') as f:
            pickle.dump(exists_list, f)

        print(data['name'], len(data['vocab']), count)
    if exists:
        return wv_matrix, exists_list
    return wv_matrix


def load_kscv(data):
    kcv = load_kcv(data)[:-2]
    kcv = np.concatenate([kcv[:, :3], kcv[:, 3:].max(1).reshape(-1, 1)], 1)
    scv = load_scv(data, normal=False)[:-2]
    assert(len(kcv[0]) == len(scv[0]))
    alpha = 150
    print('kcv alpha', alpha)
    kcv = kcv * alpha
    denominator = kcv.sum(1) + scv.sum(1)
    kscv = (kcv + scv) / denominator[:, None]
    kscv[np.isnan(kscv)] = 0
    kscv = kscv.tolist()
    dim = len(data['classes'])
    kscv.append(np.random.uniform(0, 1, dim).astype('float32'))
    kscv.append(np.zeros(dim).astype('float32'))
    kscv = np.array(kscv)
    return kscv


def get_similarity(synsets, c, ic):
    matrix = []
    for ci in c:
        temp = []
        for si in synsets:
            try:
                temp.append(wn.lin_similarity(si, ci, ic))
            except:
                temp.append(0)
        matrix.append(temp)
    return matrix


def load_kcv(data, complement=False):
    complement = (complement == '1')
    path = 'temp/{}_{}kcv.pkl'.format(data['name'], str(complement))
    if os.path.exists(path):
        print('using cache {}_{}'.format(data['name'], str(complement)))
        with open(path, 'rb') as f:
            kcv = pickle.load(f)
    else:
        dim = len(data['labels'])
        kcv = []
        class_synset = [wn.synsets(c) if len(wn.synsets(c)) != 0 else None for c in data['labels']]
        print(class_synset)
        count = 0
        w2vkcv, exists_list = load_w2vkcv(data, exists=True)
        print(len(w2vkcv[0]))
        semcor_ic = wordnet_ic.ic('ic-semcor.dat')
        for i in tqdm(range(len(data['vocab']))):
            word = data['idx_to_word'][i]
            synsets = wn.synsets(word)
            if synsets is None or len(synsets) == 0:
                if complement and exists_list[i] == 1:
                    kcv.append(w2vkcv[i])
                else:
                    kcv.append(np.random.uniform(0, 1, dim).astype('float32'))
                    count += 1
            else:
                kcv_one = []
                for c in class_synset:
                    if c is not None:
                        path_matrix = get_similarity(synsets, c, semcor_ic)
                        path_matrix = [[x if x is not None else 0 for x in y] for y in path_matrix]
                        kcv_one.append(np.max(path_matrix))
                    else:
                        kcv_one.append(np.random.random(1)[0])
                kcv.append(kcv_one)
        print(count)
        kcv.append(np.random.uniform(0, 1, dim).astype('float32'))
        kcv.append(np.zeros(dim).astype('float32'))
        kcv = np.array(kcv)
        print(kcv.shape)
        with open(path, 'wb') as f:
            pickle.dump(kcv, f)
    return kcv


def load_w2vkcv(data, exists=False):
    # agkcv = []
    # with open('data/AG/ag_vector.json', 'r') as f:
    #     ag_vector = json.load(f)
    # for i in range(len(data['vocab'])):
    #     word = data['idx_to_word'][i]
    #     if word in ag_vector.keys():
    #         agkcv.append(ag_vector[word])
    #     else:
    #         agkcv.append(np.random.uniform(0, 1, len(data['classes'])).astype('float32'))
    # agkcv.append(np.random.uniform(0, 1, len(data['classes'])).astype('float32'))
    # agkcv.append(np.zeros(len(data['classes'])).astype('float32'))
    # agkcv = np.array(agkcv)
    path = 'temp/' + data['name'] + 'w2vkcv.pkl'
    list_path = 'temp/' + data['name'] + 'w2v_list.pkl'
    if os.path.exists(path):
        print('use cache {} w2vkcv'.format(data['name']))
        with open(path, 'rb') as f:
            agkcv = pickle.load(f)
        with open(list_path, 'rb') as f:
            exists_list = pickle.load(f)
    else:
        classes = data['labels']
        dim = len(classes)
        # word_vectors = KeyedVectors.load_word2vec_format('E:/deeplearning/w2v/GoogleNews-vectors-negative300.bin', binary=True)
        word_vectors = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
        # word_vectors = KeyedVectors.load_word2vec_format('/home/cike/deeplearning/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        w2v, exists_list = load_w2v(data, exists=True)
        classes = [word_vectors.word_vec(i) for i in classes]
        # with open('temp/AGsensembed.pkl', 'rb') as f:
        #     w2v = pickle.load(f)
        # with open('temp/AGlabel_sensembed.pkl', 'rb') as f:
        #     classes = pickle.load(f)
        # dim = len(classes)
        agkcv = sp.distance.cdist(w2v[:-2], np.array(classes), 'cosine').tolist()
        agkcv.append(np.random.uniform(0, 1, dim).astype('float32'))
        agkcv.append(np.zeros(dim).astype('float32'))
        agkcv = np.array(agkcv)
        with open(path, 'wb') as f:
            pickle.dump(agkcv, f)
    if exists:
        return agkcv, exists_list
    return agkcv


def load_senticv(data, max_feature=None):
    dim = 3

    mrkcv = []
    for i in range(len(data['vocab'])):
        word = data['idx_to_word'][i]
        sentisets = list(swn.senti_synsets(word))
        if len(sentisets) > 0:
            # mrkcv.append([sentisets[0].neg_score() + sentisets[0].obj_score() / 2.0, sentisets[0].pos_score() + sentisets[0].obj_score() / 2.0])
            mrkcv.append([np.mean([s.neg_score() for s in sentisets]), np.mean([s.obj_score() for s in sentisets]), np.mean([s.pos_score() for s in sentisets])])
        else:
            mrkcv.append(np.random.uniform(0, 1, dim).astype('float32'))
    if max_feature is not None:
        mrkcv = get_max_feature(mrkcv, data, max_feature, 'np.random.uniform(0, 1, dim).astype(\'float32\')', dim)
    mrkcv.append(np.random.uniform(0, 1, dim).astype('float32'))
    mrkcv.append(np.zeros(dim).astype('float32'))
    mrkcv = np.array(mrkcv)
    return mrkcv


def load_yelpfscv(data, max_feature=None):
    yelpf = YELPF()
    yelpfscv = []
    new_value_list = [2] * yelpf.len
    for i in range(len(data['vocab'])):
        word = data['idx_to_word'][i]
        if word in yelpf.scv.keys():
            yelpfscv.append(yelpf.scv[word])
        else:
            yelpfscv.append(new_value_list)

    if max_feature is not None:
        yelpfscv = get_max_feature(yelpfscv, data, max_feature, str(new_value_list))
    yelpfscv.append(new_value_list)
    yelpfscv.append(np.zeros(yelpf.len).astype('float32'))
    yelpfscv = np.array(yelpfscv)
    return yelpfscv


def load_AGscv(data, max_feature=None):
    ag = AG(max_feature)
    agscv = []
    new_value_list = [2] * ag.len
    for i in range(len(data['vocab'])):
        word = data['idx_to_word'][i]
        if word in ag.scv.keys():
            agscv.append(ag.scv[word])
        else:
            agscv.append(new_value_list)

    agscv.append(new_value_list)
    agscv.append(np.zeros(ag.len).astype('float32'))
    agscv = np.array(agscv)
    return normalize(agscv)


class YELPF(object):
    def __init__(self):
        self.path = 'temp/yelpf_scv' + '.pkl'
        if os.path.exists(self.path):
            print('use cache yelpfscv')
            with open(self.path, 'rb') as f:
                self.scv = pickle.load(f)
        else:
            self.data = utils.get_dataset('yelpf')
            self.scv = load_scv(self.data)
            self.scv = {w: v for w, v in zip(self.data['vocab'], self.scv)}
            with open(self.path, 'wb') as f:
                pickle.dump(self.scv, f)
        self.len = len(list(self.scv.values())[0])


class AG(object):
    def __init__(self, max_feature):
        self.path = 'temp/ag_scvPARAM' + str(max_feature) + '.pkl'
        if os.path.exists(self.path):
            print('use cache AGscv')
            with open(self.path, 'rb') as f:
                self.scv = pickle.load(f)
        else:
            self.data = utils.get_dataset('AG')
            self.scv = load_scv(self.data, max_feature)
            self.scv = {w: v for w, v in zip(self.data['vocab'], self.scv)}
            with open(self.path, 'wb') as f:
                pickle.dump(self.scv, f)
        self.len = len(list(self.scv.values())[0])
