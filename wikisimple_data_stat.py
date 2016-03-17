def load_sents(path):
    sents = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for r in reader:
            sents.append(r[2])
    return sents


if __name__ == '__main__':
    import csv
    import os
    from gensim import corpora, models
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                                            level=logging.INFO)
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-input', dest='input_folder',
                           default='/home/cul226/Downloads/sentence-aligned.v2'
                           )
    argparser.add_argument('-w2v', dest='w2v')
    args = argparser.parse_args()

    # load w2v model
    model = models.Word2Vec.load_word2vec_format(args.w2v, binary=True)
    m = model.vector_size  # dimension of embedding

    norm_path = os.path.join(args.input_folder, 'normal.aligned')
    simple_path = os.path.join(args.input_folder, 'simple.aligned')

    norm_sents = load_sents(norm_path)
    not_in_w2v = set()
    for sent in norm_sents:
        wds = sent.split(' ')
        for wd in wds:
            if wd not in model:
                not_in_w2v.add(wd)
    
    print 'total {} words not in word2vec: {}'.format(len(not_in_w2v), args.w2v)
    for wd in not_in_w2v:
        print wd

