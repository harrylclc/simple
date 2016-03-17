'''
The script will prepare the Wikipedia corpus in a format that can be processed
by Mikolov's word2vec.

Before running this script, we should use WikiExtractor to extract plaintext
from the original Wikipedia xml dump.
'''
import argparse
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import multiprocessing


def repl_html_tag(s):
    return re.sub('<[^>]*>', '', s)

def process_files(queue, files):
    pid = multiprocessing.current_process().pid
    output = '{}.{}'.format(args.output, pid)
    out = open(output, 'w')
    for k, f in enumerate(files):
        if k == max_files and args.debug:
            break
        content = open(f).read()
        m = re.findall(('<doc id="(.+?)" url="(.+?)" title="(.+?)">\n'
                       '(.+?)</doc>'), content, re.S)
        for d in m:
            text = d[3]
            idx = text.find('\n\n')
            if idx != -1:
                text = text[idx+2:]
            text = repl_html_tag(text).strip().decode('utf8')
            if len(text) == 0:
                continue
            for line in text.split('\n'):
                sents = sent_tokenize(line)
                for sent in sents:
                    wds = word_tokenize(sent)
                    if len(wds) < 6:
                        continue
                    out.write(' '.join(wds).encode('utf8') + '\n')
        if k % 50 == 0:
            print 'pid:{}\t{}/{}'.format(pid, k, len(files))
    queue.put(None)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-folder', dest='extract_folder', required=True)
    argparser.add_argument('-output', dest='output', required=True)
    argparser.add_argument('-max', dest='max_files', type=int, default=1)
    argparser.add_argument('-nproc', dest='nproc', type=int, default=1)
    argparser.add_argument('--debug', action='store_true')
    args = argparser.parse_args()
    print args
    extract_folder = args.extract_folder
    nproc = args.nproc
    max_files = args.max_files
    cnt = 0
    fs = []
    for root, dirs, files in os.walk(extract_folder):
        for f in files:
            fs.append(os.path.join(root, f))
    print 'total {} files'.format(len(fs))
    arg_list = [[] for i in xrange(nproc)]
    for k, f in enumerate(fs):
        arg_list[k % nproc].append(f)

    queue = multiprocessing.Queue()
    procs = []
    for i in xrange(nproc):
        p = multiprocessing.Process(target=process_files, 
                                    args=(queue, arg_list[i]))
        procs.append(p)
        p.start()

    finished = 0
    while finished < nproc:
        r = queue.get()
        if r is None:
            finished += 1

    for p in procs:
        p.join()

