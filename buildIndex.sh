### tokenize corpus
# input: the Robust2004 corpus
#		~/documents/ir/Robust2004/corpus/
# output: ~/Documents/ir/Robust2004/result/tokenize/

#python buildIndex_tokenize.py

### extract the document infomation from the tokenized corpus
# output to ~/Documents/ir/Robust2004/result/index/doc.extract
# transform words into term_ids
# format: [doc_id \t term_id \t term_tf \t positions_in_doc]

#python buildIndex_extract.py

### sort the second column (term_id) in /doc.extract by number order
#sort -k2 -n < ~/Documents/ir/Robust2004/result/index/doc.extract > ~/Documents/ir/Robust2004/result/index/doc.extract.sort_by_termid

### build the forward index for corpus
# input: /doc.extract
# output to leveldb files: /forward_index_db

#python buildIndex_forward.py

### build the inversed index for corpus
# input: /doc.extract.sort_by_termid
# output to leveldb files: /inversed_index_db

#python buildIndex_inversed.py

### get document length
# input: /forward_index_db
#        /docname
# output:/doclen

#python buildIndex_doclen.py

### get term's document frequency

#python buildIndex_termdf.py

### apply to information retrieval
# BM25 and Language model
# input: index files
# 		 trec topics: 601-700 
# output: mean ap

python retrieval.py

./trec_eval -q /home/zxz/Documents/ir/Robust2004/qrels /home/zxz/Documents/ir/Robust2004/result/results_file > /home/zxz/Documents/ir/Robust2004/result/results_file.eval






