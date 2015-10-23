import os
import sys
import time
from bs4 import BeautifulSoup
import nltk
import leveldb

root_dir = '/home/zxz/Documents/ir/Robust2004/result/'
index_dir = root_dir + '/index/'
stopword_path = '/home/zxz/Documents/ir/Robust2004/stop-word-list'
leveldb_path = index_dir + 'forward_index_db/'
docname_path = index_dir + 'docname'
doclen_path = index_dir + 'doclen'

forward_db = leveldb.LevelDB(leveldb_path)

fo = open(docname_path, 'r')
fout = open(doclen_path,'wt')
start_time = time.time()
doc_count = 0
for line in fo:
	llist = line.strip('\n').split('\t')
	try:
		doc_info = forward_db.Get(llist[1])
	except KeyError:
		print 'Error: key not found in leveldb'
		continue
	
	doc_count += 1
	doc_len = 0
	list_term_info = doc_info.split('#')
	for term_info in list_term_info:
		tlist = term_info.split(':')
		term_tf = int(tlist[1])
		doc_len += term_tf
	
	fout.write('%s\t%s\t%d\n' %(llist[0],llist[1],doc_len))
	if doc_count % 1000 == 0:
		end_time = time.time()
		print '%f sec: #%d doc processed' %(end_time-start_time, doc_count)

fo.close()
fout.close()
