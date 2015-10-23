from __future__ import division
import os
import sys
import time
from bs4 import BeautifulSoup
import nltk
import leveldb
import math

def loadIndex(index_dir):
	global doc_names
	global doc_ids
	global doc_lens
	global term_strs
	global term_ids
	global term_dfs
	global term_ctfs
	global db_forward
	global db_inverted
	### load document names and document length
	fo = open(index_dir+'/doclen','r')
	doc_names = {}
	doc_ids = {}
	doc_lens = {}
	start_time = time.time()
	for line in fo:
		llist = line.strip('\n').split('\t')
		if len(llist) < 3:
			continue
		doc_name = llist[0]
		doc_id = int(llist[1])
		doc_len = int(llist[2])
		doc_names[doc_id] = doc_name
		doc_ids[doc_name] = doc_id
		doc_lens[doc_id] = doc_len
	end_time = time.time()
	print '%f sec: load %d document length' %(end_time-start_time,len(doc_lens))
	fo.close()
		
	### load vocab and term's document frequence
	fo = open(index_dir + '/termDocFreq','r')
	term_strs = {}
	term_ids = {}
	term_dfs = {}
	term_ctfs = {}
	start_time = time.time()
	for line in fo:
		llist = line.strip('\n').split('\t')
		if len(llist) < 4:
			continue
		term_str = llist[0]
		term_id = int(llist[1])
		term_df = int(llist[2])
		term_ctf = int(llist[3])
		term_strs[term_id] = term_str
		term_ids[term_str] = term_id
		term_dfs[term_id] = term_df
		term_ctfs[term_id] = term_ctf
	end_time = time.time()
	print '%f sec: load %d term df' %(end_time-start_time, len(term_dfs))
	fo.close()
	
	### load the leveldb files
	start_time = time.time()
	db_forward = leveldb.LevelDB(index_dir+'/forward_index_db')
	db_inverted = leveldb.LevelDB(index_dir+'/inverted_index_db')
	end_time = time.time()
	print '%f sec: load leveldb for forward and inverted index' %(end_time-start_time)

def loadStopword(stopword_path):
	global stemmer
	stop_word_list = []
	fo = open(stopword_path, 'r')
	for line in fo:
		line = line.strip()
		stop_word_list.append(line)
	stop_word_list = [stemmer.stem(word) for word in stop_word_list]
	fo.close()
	return stop_word_list

def loadGoldenTruth(golden_true_path):
	golden_truth = {}
	fo = open(golden_true_path)
	pre_query_id = -1
	doc_list = []
	start_time = time.time()
	for line in fo:
		llist = line.strip('\n').split(' ')
		if len(llist) < 4 or int(llist[3]) <= 0:
			continue
		query_id = int(llist[0])
		if pre_query_id != query_id:
			if pre_query_id == -1:
				doc_list.append(llist[2])
				pre_query_id = query_id
			else:
				golden_truth[pre_query_id] = doc_list
				doc_list = [llist[2]]
				pre_query_id = query_id
		else:
			doc_list.append(llist[2])
	golden_truth[query_id] = doc_list
	end_time = time.time()
	print '%f sec: load golden truth for %d query' %(end_time-start_time, len(golden_truth))
	fo.close()
	return golden_truth

def loadTopics(topic_desc_path):
	topics = {}
	fo = open(topic_desc_path,'r')
	start_time = time.time()
	soup = BeautifulSoup(fo,'lxml')
	for topic_i in soup.find_all('top'):
		topic_id = int(topic_i.num.text.strip().split(':')[1])
		#if topic_id < 601:
		#	continue
		topic_title = topic_i.title.text.strip()
		topics[topic_id] = topic_title
	end_time = time.time()
	print '%f sec: load %d topics' %(end_time-start_time, len(topics))
	fo.close()
	return topics

def BM25(result_file_path,topics,top_n,k1,b):
	global doc_names
	global doc_ids
	global doc_lens
	global term_strs
	global term_ids
	global term_dfs
	global db_forward
	global db_inverted
	global stemmer
	
	num_query = len(topics)
	num_doc = len(doc_names)
	avg_dlen = sum(doc_lens.values())/num_doc
	result = {}
	fo = open(result_file_path,'wt')
	for (query_id, query_title) in topics.items():
		print 'processing query %d' %(query_id)
		query_word_list = nltk.word_tokenize(query_title)
		query_term_list = [stemmer.stem(word.lower()) for word in query_word_list]
		if '' in query_term_list:
			query_term_list.remove('')
		
		query_termid_list = [term_ids[term] for term in query_term_list if term in term_ids]
		
		score_match = {}
		for query_term_id in query_termid_list:
			query_term_df = term_dfs[query_term_id]
			query_term_idf = math.log((num_doc-query_term_df+0.5)/(query_term_df+0.5))
			
			try:
				term_info = db_inverted.Get(str(query_term_id))
			except KeyError:
				print 'Error: query term not found in vocab'
				continue
			
			term_doc_list = term_info.split('#')
			for doc_i in term_doc_list:
				llist = doc_i.split(':')
				doc_id = int(llist[0])
				doc_len = doc_lens[doc_id]
				query_term_dtf = int(llist[1])
				query_term_tf = query_term_dtf * (k1 + 1) / \
							(query_term_dtf + k1 * (1-b+b*(doc_len/avg_dlen)))
				if doc_id in score_match:
					score_match[doc_id] += query_term_idf * query_term_tf
				else:
					score_match[doc_id] = query_term_idf * query_term_tf
		score_match = sorted(score_match.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
		if len(score_match) > top_n:
			score_match = score_match[:top_n]
		result[query_id] = score_match
		rank_count = 0
		for (doc_i,score) in score_match:
			rank_count += 1
			fo.write('%d Q0 %s %d %f zxz\n' %(query_id,doc_names[doc_i],rank_count, score))
	fo.close()
	return result

def LMWithSmoothing(result_file_path,topics,top_n,mu_smooth):
	global doc_names
	global doc_ids
	global doc_lens
	global term_strs
	global term_ids
	global term_dfs
	global term_ctfs
	global db_forward
	global db_inverted
	global stemmer
	
	num_query = len(topics)
	num_doc = len(doc_names)
	avg_dlen = sum(doc_lens.values())/num_doc
	total_ctf = sum(term_ctfs.values())
	result = {}
	fo = open(result_file_path,'wt')
	for (query_id, query_title) in topics.items():
		print 'processing query %d' %(query_id)
		query_word_list = nltk.word_tokenize(query_title)
		query_term_list = [stemmer.stem(word.lower()) for word in query_word_list]
		if '' in query_term_list:
			query_term_list.remove('')
		
		query_termid_list = [term_ids[term] for term in query_term_list if term in term_ids]
		
		score_match = {}
		
		term_info_dict = {}
		candidate_docs = {}
		for query_term_id in query_termid_list:
			try:
				term_info = db_inverted.Get(str(query_term_id))
			except KeyError:
				print 'Error: query term not found in vocab'
				continue
			term_info_dict[query_term_id] = term_info
			term_doc_list = term_info.split('#')
			for doc_i in term_doc_list:
				llist = doc_i.split(':')
				doc_id = int(llist[0])
				candidate_docs[doc_id] = 0
				score_match[doc_id] = 0
		candidate_docs = candidate_docs.keys()
		
		for (query_term_id,term_info) in term_info_dict.items():
			query_term_ctf = term_ctfs[query_term_id]
			query_term_prob_c = query_term_ctf / total_ctf
			
			term_doc_list = term_info.split('#')
			term_dtfs = {}
			for doc_i in term_doc_list:
				llist = doc_i.split(':')
				doc_id = int(llist[0])				
				query_term_dtf = int(llist[1])
				term_dtfs[doc_id] = query_term_dtf
			
			dtf_list = []
			dlen_list = []
			for doc_id in candidate_docs:
				dlen_list.append(doc_lens[doc_id])
				if doc_id in term_dtfs:
					dtf_list.append(term_dtfs[doc_id])
				else:
					dtf_list.append(0)					
				
			query_term_prob_d = [(dtf_list[i] + mu_smooth*query_term_prob_c)  / \
							(mu_smooth + dlen_list[i]) for i in range(len(candidate_docs))]
			
			for doc_i in range(len(candidate_docs)):
				score_match[candidate_docs[doc_i]] += math.log(query_term_prob_d[doc_i])

		score_match = sorted(score_match.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
		if len(score_match) > top_n:
			score_match = score_match[:top_n]
		print 'query: %d --- return %d docs' %(query_id,len(score_match))
		result[query_id] = score_match
		
		rank_count = 0
		for (doc_i,score) in score_match:
			rank_count += 1
			fo.write('%d Q0 %s %d %f zxz\n' %(query_id,doc_names[doc_i],rank_count, score))
		
	fo.close()
	return result
				
def evaluation(result,golden_truth):
	global doc_names
	global doc_ids
	mean_ap = []
	for (query_id,score_match) in result.items():
		if query_id not in golden_truth:
			continue
		judge_relevant = golden_truth[query_id]
		judge_relevant = [doc_ids[doc_name] for doc_name in judge_relevant]
		query_ap = []
		rank_count = 0
		rele_doc_count = 0
		for (doc_id,doc_score) in score_match:
			rank_count += 1
			if doc_id in judge_relevant:
				 rele_doc_count += 1
				 query_ap.append(rele_doc_count / rank_count)
		if len(query_ap) == 0:
			query_ap = 0
		else:
			query_ap = sum(query_ap) / len(query_ap)
		mean_ap.append(query_ap)
	return mean_ap

global doc_names
global doc_ids
global doc_lens
global term_strs
global term_ids
global term_dfs
global db_forward
global db_inverted
global stemmer

doc_names = {}
doc_ids = {}
doc_lens = {}
term_strs = {}
term_ids = {}
term_dfs = {}
db_forward = ''
db_inverted = ''

index_dir = '/home/zxz/Documents/ir/Robust2004/result/index/'
golden_true_path = '/home/zxz/Documents/ir/Robust2004/qrels'
topic_desc_path = '/home/zxz/Documents/ir/Robust2004/topics'
stopword_path = '/home/zxz/Documents/ir/Robust2004/stop-word-list'
result_dir = '/home/zxz/Documents/ir/Robust2004/result/'
result_file_path = result_dir + 'results_file'
mean_ap_path = result_dir + 'map.eval'
stemmer = nltk.stem.porter.PorterStemmer()

loadIndex(index_dir)
loadStopword(stopword_path)
golden_truth = loadGoldenTruth(golden_true_path)
topics = loadTopics(topic_desc_path)
#result = BM25(result_file_path, topics,top_n=1000,k1=0.8,b=0.75)
result = LMWithSmoothing(result_file_path,topics,top_n=1000,mu_smooth=500)
mean_ap = evaluation(result,golden_truth)
fo = open(mean_ap_path,'wt')
fo.write('%s' %('\n'.join([str(ap) for ap in mean_ap])))
fo.close()
print sum(mean_ap) / len(mean_ap)
	
	
	
	
	
	
	
	
