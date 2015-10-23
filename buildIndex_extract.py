import os
import sys
import time
from bs4 import BeautifulSoup
import nltk
import leveldb

root_dir = '/home/zxz/Documents/ir/Robust2004/result/'
data_dir = root_dir + '/tokenize/'
index_dir = root_dir + '/index/'
stopword_path = '/home/zxz/Documents/ir/Robust2004/stop-word-list'
extract_path = index_dir + 'doc.extract'
vocab_path = index_dir + 'vocab'
docname_path = index_dir + 'docname'
stemmer = nltk.stem.porter.PorterStemmer()

### get all file pathes in corpus
list_file = []
list_dir = os.walk(data_dir)
for root, dirs, files in list_dir:
	for f in files:
		list_file.append(os.path.join(root,f))
sys.stdout.write('There are #%d files to be processed.\n'%(len(list_file)))

### load stopwordlist
stop_word_list = []
fo = open(stopword_path, 'r')
for line in fo:
	line = line.strip()
	stop_word_list.append(line)
stop_word_list = [stemmer.stem(word) for word in stop_word_list]
fo.close()

### for each file
count = 0
start_time = time.time()
vocab_ids = {}
vocab_strs = {}
doc_ids = {}
doc_names = {}

vocab_count = 0
doc_count = 0

fout = open(extract_path,'wt')
for file_name in list_file:
	count += 1
	print file_name
	fo = open(file_name,'r')
	soup = BeautifulSoup(fo,'lxml')
	
	batch = leveldb.WriteBatch()
	for doc_i in soup.find_all('doc'):
		doc_ids[doc_i.docno.text.strip()] = doc_count
		doc_names[doc_count] = doc_i.docno.text.strip()
		doc_id = doc_count
		doc_count += 1
		
		sent_list = doc_i.doctext.text.split('\n')
		
		doc_index = {}
		pos_i = 0
		for sent in sent_list:
			word_list = sent.split('\t')
			for i in range(len(word_list)):
				if word_list[i] in stop_word_list:
					continue
					
				term_i = word_list[i]
				if term_i not in vocab_ids:
					vocab_ids[term_i] = vocab_count
					vocab_strs[vocab_count] = term_i
					vocab_count += 1
				
				term_id = vocab_ids[term_i]
				if term_id in doc_index:
					doc_index[term_id].append(i+pos_i)
				else:
					doc_index[term_id] = [i+pos_i]
			pos_i += len(word_list)
		for (k,v) in doc_index.items():
			fout.write('%d\t%d\t%d\t%s\n' % (doc_id,k,len(v),','.join([str(i) for i in v])))
	end_time = time.time()
	print 'time: %f sec, Processed %d docs' % (end_time - start_time, doc_count)
	fo.close()
fout.close()

fout = open(vocab_path,'wt')
for (k,v) in vocab_ids.items():
	fout.write('%s\t%d\n' %(k,v))
fout.close()
fout = open(docname_path,'wt')
for (k,v) in doc_ids.items():
	fout.write('%s\t%d\n' %(k,v))
fout.close()


