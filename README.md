# ir-python
A python implementation for information retrieval tasks, including forward/inverted index, basic retrieval models (e.g., BM25, uni-gram language model). The indexing module use a thread-safe Python bindings for LevelDB, a fast key-value storage library (https://code.google.com/p/py-leveldb/).

run: sh buildIndex.sh

1. tokenize corpus: buildIndex_tokenize.py
input: the Robust2004 corpus
output: ~/Documents/ir/Robust2004/result/tokenize/

2. extract the document infomation from the tokenized corpus: buildIndex_extract.py
input: tokenized corpus
output: transform words into term_ids to file /doc.extract
        format: [doc_id \t term_id \t term_tf \t positions_in_doc]

3. sort the second column (term_id) in /doc.extract by number order
sort -k2 -n < ./index/doc.extract > ./index/doc.extract.sort_by_termid

4. build the forward index for corpus: buildIndex_forward.py
input: /doc.extract
output to leveldb files: /forward_index_db

5. build the inversed index for corpus: buildIndex_inversed.py
input: /doc.extract.sort_by_termid
output to leveldb files: /inverted_index_db

6. get document length: buildIndex_doclen.py
input: /forward_index_db
        /docname
output:/doclen

7. get term's document frequency: buildIndex_termdf.py
input: /inverted_index_db
        /vocab
output: /termDocFreq

8. apply to information retrieval (BM25 and Language model): retrieval.py
input: index files
 		 corpus: Robust2004, trec topics: 601-700 
output:  ranking results
      /results_file

9. evaluation by trec_eval (version 9.0)
./trec_eval -q ./Robust2004/qrels ./Robust2004/result/results_file > ./Robust2004/result/results_file.eval





