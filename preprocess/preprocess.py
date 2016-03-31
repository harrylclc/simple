import argparse
import numpy as np
import os
import operator
import h5py
import re

FILE_PATHS = {'newsela': '/data/home/cul226/simple/sents/newsela.sents',
              'acl': '/data/home/cul226/simple/sents/acl.sents',
              'naacl': '/data/home/cul226/simple/sents/naacl.sents',
              'pwkp':'/data/home/cul226/simple/sents/pwkp.sents',
              'combine': '/data/home/cul226/simple/combine.sents',
             }

def remove_digits(parse):
    return re.sub(r'\d', '#', parse)


def get_vocab(dataset, path):
    max_sent_len = [0, 0]
    w2idx = {}
    w2idx['*EOS*'] = 1
    freq = {}

    with open(path) as f:
        for line in f:
            sents = line.strip().split('\t')[:2]
            for k, s in enumerate(sents):
                s = remove_digits(s)
                wds = [wd for wd in s.split(' ') if len(wd) > 0]
                max_sent_len[k] = max(max_sent_len[k], len(wds))
                for wd in wds:
                    freq[wd] = freq.get(wd, 0) + 1
                    if not wd in w2idx:
                        w2idx[wd] = len(w2idx) + 1

    if args.add_ukt:
        w2idx_filter = {}
        w2idx_filter['*EOS*'] = 1
        w2idx_filter['*UKT*'] = 2
        sorted_freq = sorted(freq.items(), key=operator.itemgetter(1),
                             reverse=True)
        for k, (wd, cnt) in enumerate(sorted_freq):
            if k == args.max_vocab_size or cnt < args.min_freq:
                break
            w2idx_filter[wd] = len(w2idx_filter) + 1
        w2idx = w2idx_filter
    return max_sent_len, w2idx, freq


def load_data(dataset, train_path):
    max_sent_len, w2idx, freq = get_vocab(dataset, train_path)
    print max_sent_len
    max_y_len = min(max_sent_len[1], args.max_len)
    data = [[], [], []]
    with open(train_path) as f:
        for line in f:
            sents = line.strip().split('\t')[:2]
            sents_id = []
            for k, s in enumerate(sents):
                s = remove_digits(s)
                wds = [wd for wd in s.split(' ') if len(wd) > 0]
                d = [w2idx[wd] if wd in w2idx else 2 for wd in wds]
                d.extend([1])   # EOS
                sents_id.append(d)
            if len(sents_id[0]) > args.max_len or\
               len(sents_id[1]) > args.max_len:
                continue
            data[2].append(len(sents_id[1]))
            sents_id[1].extend([1] * (max_y_len + 1 - len(sents_id[1])))
            for i in xrange(2):
                data[i].append(sents_id[i])
    print 'Total {} pairs of sentence'.format(len(data[0]))
    # sort input sent by len
    x_sorted = sorted(enumerate(data[0]), key=lambda x: len(x[1]))
    chunks = {}
    for kv in x_sorted:
        l = len(kv[1])
        if l not in chunks:
            chunks[l] = [[], [], []]
        chunks[l][0].append(kv[1])
        chunks[l][1].append(data[1][kv[0]])
        chunks[l][2].append(data[2][kv[0]])

    return w2idx, chunks


def load_w2v(path_to_bin, vocab):
    print 'loading {}...'.format(path_to_bin)
    word_vecs = {}
    with open(path_to_bin, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    print 'load w2v done'
    return word_vecs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', dest='data', required=True)
    parser.add_argument('-w2v', dest='w2v', required=True)
    parser.add_argument('-output_dir', dest='output_dir',
                        default='/data/home/cul226/simple/preprocessed/')
    parser.add_argument('--add_ukt', action='store_true')
    parser.add_argument('-min_freq', dest='min_freq', type=int, default=5)
    parser.add_argument('-max_vocab_size', dest='max_vocab_size', type=int,
                        default=40000)
    parser.add_argument('-max_len', dest='max_len', type=int, default=100)
    args = parser.parse_args()
    dataset = args.data
    output_dir = args.output_dir
    if dataset not in FILE_PATHS:
        print 'available datasets: {}'.format(FILE_PATHS.keys())
        exit()

    train_path = FILE_PATHS[dataset]
    w2idx, chunks = load_data(dataset, train_path)

    # save vocab
    with open(os.path.join(output_dir, dataset + '.vocab'), 'w') as out:
        for wd, idx in sorted(w2idx.items(), key=operator.itemgetter(1)):
            out.write('{} {}\n'.format(wd, idx))

    V = len(w2idx)
    print 'Vocab size:', V

    w2v = load_w2v(args.w2v, w2idx)
    print 'Words in w2v:', len(w2v)
    embed = np.random.uniform(-0.25, 0.25, (V, len(w2v.values()[0])))
    embed[0] = 0
    for wd, vec in w2v.items():
        embed[w2idx[wd] - 1] = vec

    output_file = os.path.join(output_dir, dataset + '.hdf5')
    print 'save to', output_file
    with h5py.File(output_file, 'w') as f:
        f['w2v'] = np.array(embed)
        f['x_lens'] = np.array(chunks.keys(), dtype=np.int32)
        for l in chunks:
            f['x_' + str(l)] = np.array(chunks[l][0], dtype=np.int32)
            f['y_' + str(l)] = np.array(chunks[l][1], dtype=np.int32)
            f['ylen_' + str(l)] = np.array(chunks[l][2], dtype=np.int32)

