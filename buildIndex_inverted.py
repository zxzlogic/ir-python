import os
import sys
import time
from bs4 import BeautifulSoup
import nltk
import leveldb

root_dir = '/home/zxz/Documents/ir/Robust2004/result/'
index_dir = root_dir + '/index/'
stopword_path = '/home/zxz/Documents/ir/Robust2004/stop-word-list'
data_path = index_dir + 'doc.extract.sort_by_termid'
leveldb_path = index_dir + 'inverted_index_db/'

fin = open(data_path,'r')
db = leveldb.LevelDB(leveldb_path)
term_count = 0
pre_termid = -1
term_info = []
start_time = time.time()
for line in fin:
	llist = line.strip('\n').split('\t')
	if len(llist) < 4:
		continue
		
	term_id = int(llist[1])
	if pre_termid != term_id:
		if pre_termid != -1:
			db.Put(str(pre_termid),'#'.join(term_info))
		
		term_info = [llist[0]+':'+llist[2]]
		pre_termid = term_id
		term_count += 1
		
		if term_count % 100 == 0:
			end_time = time.time()
			print '%f sec: processed %d terms' %(end_time-start_time,term_count)
	else:
		term_info.append(llist[0]+':'+llist[2])
db.Put(str(term_id),'#'.join(term_info))
print '%f sec: processed %d terms' %(end_time-start_time,term_count)
