import os
import sys
import time
from bs4 import BeautifulSoup
import nltk
import leveldb

root_dir = '/home/zxz/Documents/ir/Robust2004/result/'
index_dir = root_dir + '/index/'
stopword_path = '/home/zxz/Documents/ir/Robust2004/stop-word-list'
leveldb_path = index_dir + 'inverted_index_db/'
docfreq_path = index_dir + 'termDocFreq'
vocab_path = index_dir + 'vocab'

forward_db = leveldb.LevelDB(leveldb_path)

fo = open(vocab_path, 'r')
fout = open(docfreq_path,'wt')
start_time = time.time()
term_count = 0
for line in fo:
	llist = line.strip('\n').split('\t')
	try:
		term_info = forward_db.Get(llist[1])
	except KeyError:
		print 'Error: key not found in leveldb'
		continue
	
	term_count += 1
	list_term_info = term_info.split('#')
	term_df = len(list_term_info)
	term_tf = 0
	for doc_str in list_term_info:
		list_doc_str = doc_str.split(':')
		term_tf += int(list_doc_str[1])
	
	fout.write('%s\t%s\t%d\t%d\n' %(llist[0],llist[1],term_df,term_tf))
	if term_count % 1000 == 0:
		end_time = time.time()
		print '%f sec: #%d term processed' %(end_time-start_time, term_count)
end_time = time.time()
print '%f sec: #%d term processed' %(end_time-start_time, term_count)
fo.close()
fout.close()
