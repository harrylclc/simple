import argparse
import numpy as np
import os
import operator
import h5py

FILE_PATHS = {'newsla': '/data/home/cul226/simple/newsla_sents',
              'wiki': '/data/home/cul226/simple/aligned-good(0.67)'
             }

def get_vocab(dataset, path):
    max_sent_len = [0, 0]
    w2idx = {}
    # starts from 2 for EOS
    idx = 2

    with open(path) as f:
        for line in f:
            sents = line.strip().split('\t')[:2]
            for k, s in enumerate(sents):
                wds = s.split(' ')
                max_sent_len[k] = max(max_sent_len[k], len(wds))
                for wd in wds:
                    if not wd in w2idx:
                        w2idx[wd] = idx
                        idx += 1
    return max_sent_len, w2idx


def load_data(dataset, train_path):
    max_sent_len, w2idx = get_vocab(dataset, train_path)
    print max_sent_len
    data = [[], [], []]
    with open(train_path) as f:
        for line in f:
            sents = line.strip().split('\t')[:2]
            for k, s in enumerate(sents):
                wds = s.split(' ')
                d = [w2idx[wd] for wd in wds]
                if k == 1:
                    data[2].append(len(d) + 1)
                    d.extend([1] * (max_sent_len[k] + 1 - len(d)))  # add EOS
                data[k].append(d)
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
        out.write('*EOS* 1\n')
        for wd, idx in sorted(w2idx.items(), key=operator.itemgetter(1)):
            out.write('{} {}\n'.format(wd, idx))

    V = len(w2idx) + 1
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

