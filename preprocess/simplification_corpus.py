import os

class NewselaCorpus(object):
    def __init__(self, folder):
        self.path = os.path.join(folder,
                'newsela_articles_20150302.aligned.sents.txt')

    def get_sent_pairs(self, use_all=True):
        pairs = []
        if use_all:
            with open(self.path) as f:
                for line in f:
                    s = line.strip().split('\t')
                    pairs.append((s[3], s[4]))
        else:
            with open(self.path) as f:
                cur_doc = ''
                sent_map = {}
                for line in f:
                    s = line.strip().split('\t')
                    docid = s[0][-4:]
                    if docid != cur_doc:
                        cur_doc = docid
                        for sent in sent_map:
                            pairs.append((sent[3:], sent_map[sent][1]))
                        sent_map = {}
                    s1 = s[1] + ' '  +s[3]
                    if s1 not in sent_map:
                        sent_map[s1] = (int(s[2][1]), s[4])
                    else:
                        if int(s[2][1]) > sent_map[s1][0]:
                            sent_map[s1] = (int(s[2][1]), s[4])
        return pairs

class PWKPCorpus(object):
    '''
    PWKP dataset
    This needs tokenization.
    '''
    def __init__(self, folder):
        self.path = os.path.join(folder, 'PWKP_108016')

    def get_sent_pairs(self):
        from nltk import word_tokenize
        pairs = []
        chunks = open(self.path).read().strip().split('\n\n')
        for k, c in enumerate(chunks):
            sents = c.split('\n')
            if len(sents) != 2:
                continue
            p = []
            for sent in sents:
                wds = [wd for wd in word_tokenize(sent.strip().decode('utf8'))]
                p.append(' '.join(wds).encode('utf8', 'ignore'))
            pairs.append(p)
            if k % 5000 == 0:
                print '{}/{}'.format(k, len(chunks))
        return pairs

class TextSimpCorpus(object):
    '''
    Kauchak ACL 2013
    '''
    def __init__(self, folder):
        self.normal = os.path.join(folder, 'normal.aligned')
        self.simple= os.path.join(folder, 'simple.aligned')

    def get_sent_pairs(self):
        pairs = []
        with open(self.normal) as f1, open(self.simple) as f2:
            while True:
                s1 = f1.readline()
                if not s1:
                    break
                s1 = s1.strip()
                s2 = f2.readline().strip()
                pairs.append((s1.split('\t')[2], s2.split('\t')[2]))
        return pairs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-data')
    args = parser.parse_args()
    data = args.data

    base_dir = '/data/home/cul226/simple/datasets/'
    output_dir = '/data/home/cul226/simple/sents/'
    
    datasets = {'newsela': (NewselaCorpus, 'newsela_data_share-20150302'),
                'pwkp': (PWKPCorpus, 'PWKP'),
                'acl': (TextSimpCorpus, 'Kauchak_acl_13')
               }
    if data not in datasets:
        print 'invalid dataset option'
        print 'available datasets', datasets.keys()
        exit()
    print 'Processing dataset {}'.format(data)
    c = datasets[data][0](os.path.join(base_dir, datasets[data][1]))
    pairs = c.get_sent_pairs()
    print 'Total {} pairs of sentence'.format(len(pairs))
    with open(os.path.join(output_dir, data + '.sents'), 'w') as out:
        for p in pairs:
            out.write('{}\t{}\n'.format(p[0], p[1]))


