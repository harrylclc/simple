if __name__ == '__main__':
    import argparse
    import operator
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', dest='data', required=True)
    parser.add_argument('-min_freq', dest='min_freq', type=int, default=10)
    args = parser.parse_args()

    # get frequency
    freq = {}
    line_cnt = 0
    with open(args.data) as f:
        for line in f:
            s = line.strip().split()
            for wd in s:
                freq[wd] = freq.get(wd, 0) + 1
            line_cnt += 1
            if line_cnt % 10000 == 0:
                print line_cnt
    print 'total {} lines'.format(line_cnt)
    print 'vocab size', len(freq)
    n_words_filtered = 0

    # save vocab
    with open(args.data + '.freq', 'w') as out:
        for wd, cnt in sorted(freq.items(), key=operator.itemgetter(1)):
            out.write('{} {}\n'.format(wd, cnt))
            if cnt >= args.min_freq:
                n_words_filtered += 1

    k = 0
    # filter words under freq threshold
    with open(args.data + '.minf_{}'.format(args.min_freq), 'w') as out, \
         open(args.data) as f:
        for line in f:
            wds = line.strip().split()
            wds = ['*UKT*' if freq[wd] < args.min_freq else wd for wd in wds]
            out.write(' '.join(wds) + '\n')
            k += 1
            if k % 10000 == 0:
                print 'write {}/{}'.format(k, line_cnt)
    print 'filtered vocab size', n_words_filtered
