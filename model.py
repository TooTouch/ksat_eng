import torch
from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer, BertModel, AutoTokenizer, AutoModel
from numpy import dot
from numpy.linalg import norm
import numpy as np

import re
import itertools
import networkx as nx

# model for sentence sequence
class NextSentencePrediction:
    def __init__(self):
            # load pretrained model and a pretrained tokenizer
            self.nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #lower needed

    def nsp(self, criterion_sentence, next_sentence, verbose = 0):
        encoded = self.tokenizer.encode_plus(criterion_sentence, text_pair=next_sentence, return_tensors='pt')

        seq_relationship_logits = model(**encoded)[0]

        # we still need softmax to convert the logits into probabilities
        # index 0: sequence B is a continuation of sequence A
        # index 1: sequence B is a random sequence
        probs = softmax(seq_relationship_logits, dim=1)
        if verbose != 0:
            print(f'{criterion_sentence} \n >>>  {next_sentence} \n')
        return probs

    def sentence_order(self, first,sentences):
        sentence = sentences.copy()
        start = first
        answer_sent = [first]
        answer_index = []
        
        while len(sentence) > 1:
            sent_dic = {}
            for s_key in sentence.keys():
                false_prob = nsp(start, sentence[s_key])[0][1]
                sent_dic[s_key] = false_prob
                
            min_prob = min(sent_dic.items(), key=lambda x: x[1])

            start = start +' '+sentence[min_prob[0]]

            answer_index.append(min_prob[0])
            answer_sent.append(sentence.pop(min_prob[0]))
            
        # last sentence append
        answer_sent.append(list(sentence.values())[0])
        answer_index.append(list(sentence.keys())[0])
        
        return answer_sent, answer_index

class SummarizationBasedModel:
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sent_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-cls-token")
        self.sent_model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-cls-token")   

    def make_sentence_graph(self, sentence, min_sim):
        sentence_graph = nx.Graph()  # initialize an undirected graph
        sentence_graph.add_nodes_from(sentence)

        nodePairs = list(itertools.combinations(sentence, 2))

        # add edges to the graph (weighted by Levenshtein distance)
        for pair in nodePairs:
            node1 = pair[0]
            node2 = pair[1]

            cos_sim = dot(sentence[pair[0]][1], sentence[pair[1]][1]) / (
                norm(sentence[pair[0]][1]) * norm(sentence[pair[1]][1])
            )

            if cos_sim > min_sim:
                sentence_graph.add_edge(node1, node2, weight=cos_sim)

        return sentence_graph


    def extract_sentence(self, sentence_graph, sentence, top_k):
        calculated_page_rank = nx.pagerank(
            sentence_graph, alpha=0.85, max_iter=100, weight="weight"
        )

        sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

        modified_sentence = sentences[: -len(sentences) + top_k]
        result_sentence = [(sentence[sent][0], sent) for sent in modified_sentence]

        return result_sentence

    def sentence_summary(self, article, min_sim, top_k, cleaned = True):

        sentence_sum_result = []

        original_sentences = [s for s in article.split('.')]

        # sentence마다의 embedding이 포함되어 있다.
        article_embs = [sentence_to_bert_sentemb(s) for s in original_sentences]
        sentence = {}

        for idx, sent in enumerate(article_embs):
            sentence[original_sentences[idx]] = [idx, sent]

        sentence_graph = make_sentence_graph(sentence, min_sim=min_sim)

        extracted_sentence = extract_sentence(sentence_graph, sentence, top_k=top_k)
        sentence_sum_result.append(extracted_sentence)
        
        if cleaned:
            sentence_sum_result = [i[1] for i in sentence_sum_result[0]]
            sentence_sum_result = '.'.join(sentence_sum_result)

        return sentence_sum_result

    # need the whold article / not a single sentence
    def sentence_summary_sentemb(article, min_sim, top_k, cleaned=True):
        sentence_sum_result = []
        original_sentences = [i.strip() for i in article.split('.')][:-1]
        
        # sentence마다의 embedding이 포함되어 있다.
        article_embs = sent_emb(original_sentences)
        sentence = {}

        for idx, sent in enumerate(article_embs):
            sentence[original_sentences[idx]] = [idx, sent]

        sentence_graph = make_sentence_graph(sentence, min_sim=min_sim)

        extracted_sentence = extract_sentence(sentence_graph, sentence, top_k=top_k)
        sentence_sum_result.append(extracted_sentence)
        
        if cleaned:
            sentence_sum_result = [i[1] for i in sentence_sum_result[0]]
            sentence_sum_result = '.'.join(sentence_sum_result)

        return sentence_sum_result

    def sentence_to_bert_emb(self, sentence):
        # turn a single sentence to sentence embedding by combined word embedding

        marked_sentence = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_sentence)
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        
        with torch.no_grad():

            outputs = self.model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on 
            # how it's  configured in the `from_pretrained` call earlier. In this case, 
            # becase we set `output_hidden_states = True`, the third item will be the 
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        
        return sentence_embedding

    def sentences_to_bert_sentemb(self, sentences):
        # Embedding Sentences in Paragraph by sentence_tokenizer and sentence_Bert
        encoded_input = self.sent_tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            model_output = self.sent_model(**encoded_input)
            sentence_embeddings = model_output[0][:,0] #Take the first token ([CLS]) from each sentence 

        return sentence_embeddings 

    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))

    def return_max(summary, answers):
        maxi = 0
        for i, sent_emb in enumerate(answers):
            similarity = cos_sim(summary, sent_emb)
            print(f'answer {i}\'s similarity with summary is {similarity}')
            
            if maxi < similarity:
                maxi = similarity

        return maxi
