import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.name = 'CNN'
        self.MODEL = kwargs['MODEL']
        self.BATCH_SIZE = kwargs['BATCH_SIZE']
        self.MAX_SENT_LEN = kwargs['MAX_SENT_LEN']
        self.WORD_DIM = kwargs['WORD_DIM']
        self.VOCAB_SIZE = kwargs['VOCAB_SIZE']
        self.CLASS_SIZE = kwargs['CLASS_SIZE']
        self.FILTERS = kwargs['FILTERS']
        self.FILTER_NUM = kwargs['FILTER_NUM']
        self.DROPOUT_PROB = kwargs['DROPOUT_PROB']
        self.SOFTMAX = True if kwargs['DATASET'] not in ['SST1', 'MR'] else False
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)

        self.WV_MATRIX = kwargs['WV_MATRIX']
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        if self.MODEL == 'static':
            self.embedding.weight.requires_grad = False
        elif self.MODEL.startswith('multichannel'):
            self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
            self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            self.embedding2.weight.requires_grad = False
            self.IN_CHANNEL = 2

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, 'conv_{}'.format(i), conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        # self.softmax = nn.Softmax(dim=1)

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        if self.MODEL == 'multichannel':
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)
        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]
        # conv_results = [
        #     F.relu(F.max_pool1d(self.get_conv(i)(x), self.MAX_SENT_LEN - self.FILTERS[i] + 1)).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)
        # if self.SOFTMAX:
        #     x = self.softmax(x)
        return x


# not used
class CNN2D(nn.Module):
    def __init__(self, **kwargs):
        super(CNN2D, self).__init__()

        self.name = 'CNN2D'
        self.MODEL = kwargs['MODEL']
        self.BATCH_SIZE = kwargs['BATCH_SIZE']
        self.MAX_SENT_LEN = kwargs['MAX_SENT_LEN']
        self.WORD_DIM = kwargs['WORD_DIM']
        self.VOCAB_SIZE = kwargs['VOCAB_SIZE']
        self.CLASS_SIZE = kwargs['CLASS_SIZE']
        self.FILTERS = kwargs['FILTERS']
        self.FILTER_NUM = kwargs['FILTER_NUM']
        self.DROPOUT_PROB = kwargs['DROPOUT_PROB']
        self.MAX_POOLING_SIZE = 2
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)

        self.WV_MATRIX = kwargs['WV_MATRIX']
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))

        for i in range(len(self.FILTERS)):
            conv = nn.Conv2d(self.IN_CHANNEL, self.FILTER_NUM[i], self.FILTERS[i])
            setattr(self, 'conv_{}'.format(i), conv)
            setattr(self, 'deep_featue_{}'.format(i), self.map_size(i))
            print(self.get_deep_feature(i))

        self.fc = nn.Linear(sum([self.get_deep_feature(i) * self.FILTER_NUM[i] for i in range(len(self.FILTERS))]), self.CLASS_SIZE)
        # self.softmax = nn.Softmax(dim=1)

    def side_size(self, src, kernel, stride=1, padding=0, dilation=1):
        return math.floor((src + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

    def map_size(self, index):
        a = self.side_size(self.side_size(self.MAX_SENT_LEN, self.FILTERS[index]), self.MAX_POOLING_SIZE, stride=self.MAX_POOLING_SIZE)
        b = self.side_size(self.side_size(self.WORD_DIM, self.FILTERS[index]), self.MAX_POOLING_SIZE, stride=self.MAX_POOLING_SIZE)
        return a * b

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def get_deep_feature(self, i):
        return getattr(self, 'deep_featue_{}'.format(i))

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.MAX_SENT_LEN, self.WORD_DIM)
        conv_results = [
            F.max_pool2d(F.relu(self.get_conv(i)(x)), self.MAX_POOLING_SIZE).view(-1, self.get_deep_feature(i) * self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)
        return x


class CNN3(nn.Module):
    def __init__(self, **kwargs):
        super(CNN3, self).__init__()

        self.name = 'CNN3'
        self.MODEL = kwargs['MODEL']
        self.BATCH_SIZE = kwargs['BATCH_SIZE']
        self.MAX_SENT_LEN = kwargs['MAX_SENT_LEN']
        self.VOCAB_SIZE = kwargs['VOCAB_SIZE']
        self.CLASS_SIZE = kwargs['CLASS_SIZE']
        self.FILTERS = kwargs['FILTERS']
        self.FILTER_NUM = kwargs['FILTER_NUM']
        self.DROPOUT_PROB = kwargs['DROPOUT_PROB']
        self.WV_MATRIX = kwargs['WV_MATRIX']
        self.WORD_DIM = kwargs['WORD_DIM']
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM[0], padding_idx=self.VOCAB_SIZE + 1)
        self.embedding_cv = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM[1], padding_idx=self.VOCAB_SIZE + 1)
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX[0]))
        self.embedding_cv.weight.data.copy_(torch.from_numpy(self.WV_MATRIX[1]))
        if self.MODEL == 'static':
            self.embedding.weight.requires_grad = False
            self.embedding_cv.weight.requires_grad = False
        elif self.MODEL.startswith('multichannel'):
            self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM[0], padding_idx=self.VOCAB_SIZE + 1)
            self.embedding2_cv = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM[1], padding_idx=self.VOCAB_SIZE + 1)
            self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX[0]))
            self.embedding2_cv.weight.data.copy_(torch.from_numpy(self.WV_MATRIX[1]))
            self.embedding2.weight.requires_grad = False
            self.embedding2_cv.weight.requires_grad = False
            self.IN_CHANNEL = 2

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM[0] * self.FILTERS[i], stride=self.WORD_DIM[0])
            setattr(self, 'conv_{}'.format(i), conv)

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM[1] * self.FILTERS[i], stride=self.WORD_DIM[1])
            setattr(self, 'conv_cv_{}'.format(i), conv)

        self.hidden = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        self.hidden_cv = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        self.fc = nn.Linear(self.CLASS_SIZE * 2, self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def get_conv_cv(self, i):
        return getattr(self, 'conv_cv_{}'.format(i))

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM[0] * self.MAX_SENT_LEN)
        xcv = self.embedding_cv(inp).view(-1, 1, self.WORD_DIM[1] * self.MAX_SENT_LEN)
        if self.MODEL == 'multichannel':
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM[0] * self.MAX_SENT_LEN)
            x2cv = self.embedding2_cv(inp).view(-1, 1, self.WORD_DIM[1] * self.MAX_SENT_LEN)

            x = torch.cat((x, x2), 1)
            xcv = torch.cat((xcv, x2cv), 1)
        # x = F.dropout(x, p=0.3, training=self.training)
        # xcv = F.dropout(xcv, p=0.3, training=self.training)
        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]

        conv_cv_results = [
            F.max_pool1d(F.relu(self.get_conv_cv(i)(xcv)), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        xcv = torch.cat(conv_cv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        xcv = F.dropout(xcv, p=self.DROPOUT_PROB, training=self.training)

        x = F.tanh(self.hidden(x))
        xcv = F.tanh(self.hidden_cv(xcv))
        # x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        # xcv = F.dropout(xcv, p=self.DROPOUT_PROB, training=self.training)
        x = torch.cat((x, xcv), 1)
        x = self.fc(x)
        # x = F.softmax(x, dim=1)
        return x


class BLSTM(nn.Module):
    def __init__(self, **kwargs):
        super(BLSTM, self).__init__()

        self.name = 'BLSTM'
        self.BATCH_SIZE = kwargs['BATCH_SIZE']
        self.MAX_SENT_LEN = kwargs['MAX_SENT_LEN']
        self.VOCAB_SIZE = kwargs['VOCAB_SIZE']
        self.CLASS_SIZE = kwargs['CLASS_SIZE']
        self.WORD_DIM = kwargs['WORD_DIM']
        self.HIDDEN_SIZE = 300
        self.LSTM_DROP_OUT = 0.2
        self.DROP_OUT = 0.4
        self.EMBEDDING_DROP_OUT = 0.5
        self.NUM_LAYERS = 1
        self.BIDIRECTIONAL = True
        self.IN_CHANNEL = 1
        self.FILTERS = 3
        self.FILTER_NUM = 100
        self.MAX_POOLING_SIZE = 2
        self.DEEP_FEATURE_SIZE = self.FILTER_NUM * self.map_size()

        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        self.WV_MATRIX = kwargs['WV_MATRIX']
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))

        self.lstm = nn.LSTM(input_size=self.WORD_DIM, batch_first=True, hidden_size=self.HIDDEN_SIZE, num_layers=self.NUM_LAYERS, bidirectional=self.BIDIRECTIONAL)
        self.conv = nn.Conv2d(self.IN_CHANNEL, self.FILTER_NUM, self.FILTERS)

        self.fc = nn.Linear(self.DEEP_FEATURE_SIZE, self.CLASS_SIZE)
        self.hidden = self.init_hidden_state()

    def init_hidden_state(self):
        return None

    def side_size(self, src, kernel, stride=1, padding=0, dilation=1):
        return math.floor((src + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

    def map_size(self):
        a = self.side_size(self.side_size(self.MAX_SENT_LEN, self.FILTERS), self.MAX_POOLING_SIZE, stride=self.MAX_POOLING_SIZE)
        b = self.side_size(self.side_size(self.HIDDEN_SIZE * 2, self.FILTERS), self.MAX_POOLING_SIZE, stride=self.MAX_POOLING_SIZE)
        return a * b

    def forward(self, inp):
        x_e = self.embedding(inp).view(-1, self.MAX_SENT_LEN, self.WORD_DIM)
        x = F.dropout(x_e, p=self.EMBEDDING_DROP_OUT, training=self.training)
        x, hidden = self.lstm(x, None)
        # x = torch.cat([x[:, :, :self.HIDDEN_SIZE], x_e, x[:, :, self.HIDDEN_SIZE:]], 2)
        x = F.dropout(x, p=self.LSTM_DROP_OUT, training=self.training)
        x = x.unsqueeze(1)

        x = F.max_pool2d(F.relu(self.conv(x)), self.MAX_POOLING_SIZE).view(-1, self.DEEP_FEATURE_SIZE)
        x = F.dropout(x, p=self.DROP_OUT, training=self.training)
        x = self.fc(x)
        # x = F.softmax(x, dim=1)
        return x


class CNNCCA(nn.Module):
    def __init__(self, **kwargs):
        super(CNNCCA, self).__init__()

        self.name = 'CNNCCA'
        self.BATCH_SIZE = kwargs['BATCH_SIZE']
        self.MAX_SENT_LEN = kwargs['MAX_SENT_LEN']
        self.VOCAB_SIZE = kwargs['VOCAB_SIZE']
        self.CLASS_SIZE = kwargs['CLASS_SIZE']
        self.WORD_DIM = kwargs['WORD_DIM']
        self.DROPOUT_PROB = kwargs['DROPOUT_PROB']
        self.WV_MATRIX = kwargs['WV_MATRIX']
        self.FILTERS = kwargs['FILTERS']
        self.FILTER_NUM = kwargs['FILTER_NUM']
        self.GPU = kwargs['GPU']
        self.idx_to_word = kwargs['idx_to_word']
        self.word_to_idx = kwargs['word_to_idx']
        self.concept_vectors = kwargs['concept_vectors']
        self.IN_CHANNEL = 1

        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        self.attend = nn.Linear(self.WORD_DIM, 1)

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i] * 2, stride=self.WORD_DIM * 2)
            setattr(self, 'conv_{}'.format(i), conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_concept(self, matrix, context):
        context_np = context.cpu().data.numpy()
        concept_matrix = []
        for index, sent in enumerate(matrix):
            concept_sent = []
            context_sent = context_np[index].reshape(1, -1)
            for word in sent:
                if int(word) < self.VOCAB_SIZE:
                    word = self.idx_to_word[int(word)].lower()
                    if word in self.concept_vectors['concept'].keys():
                        concepts = self.concept_vectors['concept'][word]
                        concepts = [self.concept_vectors['word_vectors'][cp.split()[0]] if cp.split()[0] in self.concept_vectors['word_vectors'].vocab else None for cp in concepts]
                        concepts = np.array(list(filter(lambda x: x is not None, concepts))).reshape(-1, self.WORD_DIM)
                        if concepts.shape[0] == 0:
                            concept = np.random.uniform(-0.01, 0.01, self.WORD_DIM).astype('float32')
                        else:
                            concepts_cos = cosine_similarity(concepts, context_sent)
                            concept = concepts[np.argmax(concepts_cos)]
                    else:
                        concept = np.random.uniform(-0.01, 0.01, self.WORD_DIM).astype('float32')
                elif int(word) == self.VOCAB_SIZE:
                    concept = np.random.uniform(-0.01, 0.01, self.WORD_DIM).astype('float32')
                else:
                    concept = np.zeros(self.WORD_DIM).astype('float32')
                concept_sent.append(concept)
            concept_matrix.append(concept_sent)
        concept_matrix = np.array(concept_matrix)
        return Variable(torch.from_numpy(concept_matrix)).cuda(self.GPU)

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def forward(self, inp):
        x = self.embedding(inp).view(-1, self.MAX_SENT_LEN, self.WORD_DIM)
        attn_weights = F.softmax(self.attend(x), dim=1).view(-1, 1, self.MAX_SENT_LEN)
        context = torch.bmm(attn_weights, x).view(-1, self.WORD_DIM)
        concept_matrix = self.get_concept(inp, context)
        x = torch.cat((x, concept_matrix), 2).view(-1, 1, self.MAX_SENT_LEN * self.WORD_DIM * 2)
        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]
        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)
        return x


class AttBLSTM(nn.Module):
    def __init__(self, **kwargs):
        super(AttBLSTM, self).__init__()

        self.name = 'AttBLSTM'
        self.BATCH_SIZE = kwargs['BATCH_SIZE']
        self.MAX_SENT_LEN = kwargs['MAX_SENT_LEN']
        self.VOCAB_SIZE = kwargs['VOCAB_SIZE']
        self.CLASS_SIZE = kwargs['CLASS_SIZE']
        self.WORD_DIM = kwargs['WORD_DIM']
        self.WV_MATRIX = kwargs['WV_MATRIX']
        self.HIDDEN_SIZE = self.WORD_DIM
        self.ATTN_SIZE = self.HIDDEN_SIZE
        self.NUM_LAYERS = 1
        self.BIDIRECTIONAL = True
        self.DROPOUT_PROB = [0.3, 0.3, 0.5]
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        self.bilstm = nn.LSTM(input_size=self.WORD_DIM, batch_first=True, hidden_size=self.HIDDEN_SIZE, num_layers=self.NUM_LAYERS, bidirectional=self.BIDIRECTIONAL)
        self.atten = nn.Linear(self.ATTN_SIZE, 1)
        self.fc = nn.Linear(self.ATTN_SIZE, self.CLASS_SIZE)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, self.MAX_SENT_LEN, self.WORD_DIM)
        x = F.dropout(x, p=self.DROPOUT_PROB[0], training=self.training)
        x, hidden = self.bilstm(x, None)
        x = x[:, :, :self.HIDDEN_SIZE] + x[:, :, self.HIDDEN_SIZE:]
        # x = F.tanh(x)
        x = F.dropout(x, p=self.DROPOUT_PROB[1], training=self.training)
        attn_weights = F.tanh(F.softmax(self.atten(x), dim=2)).view(-1, 1, self.MAX_SENT_LEN)
        x = torch.bmm(attn_weights, x).view(-1, self.ATTN_SIZE)
        x = F.tanh(x)
        x = F.dropout(x, p=self.DROPOUT_PROB[2], training=self.training)
        x = self.fc(x)
        return x


class RCNN(nn.Module):
    def __init__(self, **kwargs):
        super(RCNN, self).__init__()

        self.name = 'RCNN'
        self.BATCH_SIZE = kwargs['BATCH_SIZE']
        self.MAX_SENT_LEN = kwargs['MAX_SENT_LEN']
        self.VOCAB_SIZE = kwargs['VOCAB_SIZE']
        self.CLASS_SIZE = kwargs['CLASS_SIZE']
        self.WORD_DIM = kwargs['WORD_DIM']
        self.WV_MATRIX = kwargs['WV_MATRIX']
        self.HIDDEN_SIZE = 50
        self.HIDDEN_OUTPUT = 100
        self.NUM_LAYERS = 1
        self.BIDIRECTIONAL = True
        self.DROPOUT_PROB = [0.3, 0.3, 0.5]
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        self.bilstm = nn.LSTM(input_size=self.WORD_DIM, batch_first=True, hidden_size=self.HIDDEN_SIZE, num_layers=self.NUM_LAYERS, bidirectional=self.BIDIRECTIONAL)
        self.hidden = nn.Linear((self.HIDDEN_SIZE * 2 + self.WORD_DIM), self.HIDDEN_OUTPUT)
        self.fc = nn.Linear(self.HIDDEN_OUTPUT, self.CLASS_SIZE)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, self.MAX_SENT_LEN, self.WORD_DIM)
        x = F.dropout(x, p=self.DROPOUT_PROB[0], training=self.training)
        context, hidden = self.bilstm(x, None)
        x = torch.cat([context[:, :, :self.HIDDEN_SIZE], x, context[:, :, self.HIDDEN_SIZE:]], 2)
        x = F.dropout(x, p=self.DROPOUT_PROB[1], training=self.training)
        x = F.tanh(self.hidden(x))
        x = torch.max(x, 1)[0]
        x = F.dropout(x, p=self.DROPOUT_PROB[2], training=self.training)
        x = self.fc(x)
        return x
