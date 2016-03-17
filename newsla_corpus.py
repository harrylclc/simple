if __name__ == '__main__':
    f_input = ('/data/home/cul226/simple/newsela_article_corpus_2016-01-29/'
               'newsela_data_share-20150302/'
               'newsela_articles_20150302.aligned.sents.txt')
    output = '/data/home/cul226/simple/newsla_sents'
    with open(f_input) as f, open(output, 'w') as out:
        cur_doc = ''
        sent_map = {}
        for line in f:
            s = line.strip().split('\t')
            docid = s[0][-4:]
            if docid != cur_doc:
                cur_doc = docid
                for sent in sent_map:
                    out.write('{}\t{}\n'.format(sent[3:], sent_map[sent][1]))
                sent_map = {}
            s1 = s[1] + ' '  +s[3]
            if s1 not in sent_map:
                sent_map[s1] = (int(s[2][1]), s[4])
            else:
                if int(s[2][1]) > sent_map[s1][0]:
                    sent_map[s1] = (int(s[2][1]), s[4])
