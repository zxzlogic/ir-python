import os
import sys
import time
from bs4 import BeautifulSoup
import nltk
import leveldb

root_dir = '/home/zxz/Documents/ir/Robust2004/result/'
index_dir = root_dir + '/index/'
stopword_path = '/home/zxz/Documents/ir/Robust2004/stop-word-list'
data_path = index_dir + 'doc.extract'
leveldb_path = index_dir + 'forward_index_db/'

fin = open(data_path,'r')
db = leveldb.LevelDB(leveldb_path)
doc_count = 0
pre_docid = -1
doc_info = []
start_time = time.time()
for line in fin:
	llist = line.strip('\n').split('\t')
	if len(llist) < 4:
		continue
		
	doc_id = int(llist[0])
	if pre_docid != doc_id:
		if pre_docid != -1:
			db.Put(str(pre_docid),'#'.join(doc_info))
		
		doc_info = [llist[1]+':'+llist[2]+':'+llist[3]]
		pre_docid = doc_id
		doc_count += 1
		
		if doc_count % 10000 == 0:
			end_time = time.time()
			print '%f sec: processed %d doc' %(end_time-start_time,doc_count)
	else:
		doc_info.append(llist[1]+':'+llist[2]+':'+llist[3])
db.Put(str(doc_id),'#'.join(doc_info))
print '%f sec: processed %d doc' %(end_time-start_time,doc_count)
