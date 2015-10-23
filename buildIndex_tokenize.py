import os
import sys
import time
from bs4 import BeautifulSoup
import nltk
import multiprocessing
import logging
#from multiprocessing.dummy import Pool as ThreadPool

def worker_tokenize(file_path):
	global tokenize_dir
	global stemmer
	print 'Begin: ' + file_path
	filename = '_'.join(file_path.split('/'))
	fin = open(file_path,'r')
	fout = open(tokenize_dir+filename,'wt')
	doc_count = 0
	soup = BeautifulSoup(fin,'lxml')
	for doc_i in soup.find_all('doc'):
		fout.write('<doc>\n')
		fout.write('<docno> %s </docno>\n' % (doc_i.docno.text.strip()))
		doc_count += 1
		
		doc_text = ''
		if doc_i.h3 != None:
			doc_text += doc_i.h3.get_text().strip() + '.\n'
		if doc_i.h4 != None:
			doc_text += doc_i.h4.get_text().strip() + '.\n'
		for text_in_doc in doc_i.select('text'):
			doc_text += text_in_doc.get_text()
		
		fout.write('<doctext>\n')
		
		for sent in nltk.sent_tokenize(doc_text):
			word_list = nltk.word_tokenize(sent)
			word_list = [word.lower() for word in word_list \
						if ((word[0].isdigit() or word[0].isalpha()))]
			term_list = []
			
			for i in range(len(word_list)):	
				term_i = stemmer.stem(word_list[i])
				if term_i != '':
					term_list.append(term_i)
			try:
				fout.write('%s\n' % ('\t'.join(term_list)))
			except:
				for i in range(len(term_list)):
					try:
						fout.write('%s' % (term_list[i]))
						if i < len(term_list)-1:
							fout.write('\t')
					except:
						continue
				fout.write('\n')
							
		fout.write('</doctext>\n')
		fout.write('</doc>\n')
	fin.close()
	fout.close()
	print 'end: %s,%d' %(file_path,doc_count)

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)
logger.warning('doomed')

global root_dir
root_dir = '/home/zxz/Documents/ir/Robust2004/corpus/'
global tokenize_dir
tokenize_dir = '/home/zxz/Documents/ir/Robust2004/result/tokenize/'
global stemmer
stemmer = nltk.stem.porter.PorterStemmer()

### get all file pathes in corpus
list_file = []
list_dir = os.walk(root_dir)
for root, dirs, files in list_dir:
	for f in files:
		list_file.append(os.path.join(root,f))
sys.stdout.write('There are #%d files to be processed.\n'%(len(list_file)))

### for each file, tokenize plus Porter stemmer
count = 0
start_time = time.time()
# Make the Pool of workers
pool = multiprocessing.Pool(processes=6)
result = []
# Open the urls in their own threads
# and return the results
for file_i in list_file:
	pool.apply_async(worker_tokenize,(file_i, ))
	#pool.map(worker_tokenize, list_file)
#close the pool and wait for the work to finish 
pool.close() 
pool.join()
end_time = time.time()
print 'Total time: %f' %(end_time - start_time)

