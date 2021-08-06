import pickle
import nltk
import spacy
import benepar
import torch
import json
import os
import subprocess
import numpy as np
from tempfile import TemporaryDirectory
from fairseq.models.bart import BARTModel
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords 
from nltk.tree import Tree
from nltk.tree import ParentedTree
from benepar.spacy_plugin import BeneparComponent
from collections import defaultdict, Counter
import time

from transformers.pipelines import pipeline
import warnings
warnings.filterwarnings("ignore")

#from bert_score import BERTScorer
benepar.download('benepar_en2')
nltk.download('stopwords')

class FEQA(object):
    def __init__(self, device='cpu', qa_model_name = "deepset/minilm-uncased-squad2", qg_model_dir='../feqa/bart_qg/checkpoints/'):
        
        self.qg_model = BARTModel.from_pretrained(
            qg_model_dir,
            checkpoint_file = 'checkpoint_best.pt'
            )

        if device=='cuda':
            self.qg_model.to(device) #.cuda()
            self.qg_model.half()
        self.qg_model.eval()

        self.batch_size = 1#64
        self.beam_size = 10
        self.max_length = 100

        self.nlp = spacy.load('en_core_web_sm')
        #self.parser = benepar.Parser("benepar_en2")
        self.stop_words = set(stopwords.words('english'))

        self.qa_threshold = 0.1 # below threshold, the question quality is too vague
        self.qa_pipeline = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name)
        
    #    self.bertscorer = BERTScorer(lang="en") #, rescale_with_baseline=True)

    def _get_entities(self, output_summary):
        entities = [X.text for X in self.nlp(output_summary).ents]
        return entities


    # def _get_masked_phrases(self, output_summary, phrase_types=["NP"]):
    #     masked_phrases = []
    #     parse_tree = self.parser.parse(output_summary)
    #     for subtree in parse_tree.subtrees():
    #         phrases_list = [(subtree_.leaves(), subtree_.label()) for subtree_ in subtree if type(subtree_) == Tree and subtree_.label() in phrase_types]
    #         for phrase_tuple in phrases_list:
    #             phrase = phrase_tuple[0]
    #             phrase_type = phrase_tuple[1]
    #             phrase_text = " ".join(phrase)
    #             if len(phrase) > 0 and phrase_text not in self.stop_words:
    #                 masked_phrases.append(phrase_text)
       
    #     return masked_phrases 


    def _generate_questions(self, summaries, entities=True, phrase_types=["NP"]):
        doc_ids = []
        qa_masks = []
        tokenized_phrases = []

        for id_, summary in enumerate(summaries):
            summary = summary.strip()
            all_masked_phrases = []
            if entities:
                all_masked_phrases.extend(self._get_entities(summary))
            # all_masked_phrases.extend(self._get_masked_phrases(summary,phrase_types))
            all_masked_phrases = list(set(all_masked_phrases))

            for i, masked_phrase in enumerate(all_masked_phrases):
                tokenized_summary = " ".join(nltk.word_tokenize(summary.lower()))
                tokenized_phrase = " ".join(nltk.word_tokenize(masked_phrase.lower()))

                qa_masks.append(tokenized_summary + " [SEP] " + tokenized_phrase)
                doc_ids.append(str(id_))
                tokenized_phrases.append(tokenized_phrase)

        questions = []
        for i in range(0, len(qa_masks), self.batch_size):
            batch = qa_masks[i:i + self.batch_size]
            hypotheses = self.qg_model.sample(batch, beam=self.beam_size, lenpen=1.0, max_len_b=self.max_length, min_len=1, no_repeat_ngram_size=3)
            questions.extend(hypotheses)


        return doc_ids, questions, tokenized_phrases

    def _convert_to_squad_format(self, gold_answers, questions, doc_ids, bodies):
        squad_format = {"data":[]}
        
        id_questions=defaultdict(list)
        id_gold_answers=defaultdict(str)

        for idx in range(0,len(doc_ids)):
            id_questions[doc_ids[idx].strip()].append((questions[idx], gold_answers[idx]))
        
        for idx in id_questions:
            paragraphs = []
            context = bodies[int(idx)].strip()

            title = "doc_" + str(idx)
            
            questions_list_input=[]
            for q_id, question in enumerate(id_questions[idx]):

                gold_answer = question[1]
                question_text = question[0]
                answers_input = [{"text": gold_answer, "answer_start": 0}]
                questions_input = {
                                    "question": question_text, 
                                    "answers": answers_input, 
                                    "id": str(idx).strip() + "-" + str(q_id)
                                    }
                questions_list_input.append(questions_input) 
                id_gold_answers[questions_input["id"]] = gold_answer      

            
            paragraphs.append({"context":" ".join(nltk.word_tokenize(context)).lower(),"qas":questions_list_input})
            squad_format["data"].append({"title":title,"paragraphs":paragraphs})

            
        squad_format["version"] = "1.1"
        return id_gold_answers, squad_format
    
    def _answer_questions_by_context(self, squad_format):
        id_answers=defaultdict(str)

        for doc in squad_format['data']:
            for para in doc['paragraphs']:
                context = para['context']

                for q in para['qas']:
                    inputs = {
                        'question': q['question'],
                        'context': context
                    }
                    ret = self.qa_pipeline(inputs)
                    id_answers[q["id"]] = ret
    #                 print(q['question'])
    #                 print(ret)
    #             print()
        return id_answers

    def _readable_qas_dict(self, doc_ids, questions, gold_answers, pred_dict, bodies):
        qas_dict = defaultdict()
        previous_doc_id = None

        for idx, qa_id in enumerate(pred_dict):
            qa_info = {"question": questions[idx], 
                       "gold_ans": gold_answers[idx], 
                       "reply_ans": pred_dict[qa_id]["answer"],
                       "reply_scr": pred_dict[qa_id]["score"]}

#             if qa_info["reply_scr"] > self.qa_threshold:
#                 cand =  [qa_info["reply_ans"]]
#                 ref  =  [qa_info['gold_ans']]
#                 _, _, bert_f1 = self.bertscorer.score( cand, ref )
#                 qa_info["bert_f1"] = bert_f1.item()
#                 doc_f1_list.append(qa_info["bert_f1"])
#             else:
#                 qa_info["bert_f1"] = None
            
            doc_id = doc_ids[idx].strip()
            
            if doc_id != previous_doc_id:
#                 if previous_doc_id is not None:
#                     if len(doc_f1_list) == 0:
#                         qas_dict[previous_doc_id]["doc_f1"] = 0
#                     else:
#                         qas_dict[previous_doc_id]["doc_f1"] = np.mean(doc_f1_list)
#                         doc_f1_list = []
                
                qas_dict[doc_id] = dict()
                qas_dict[doc_id]['context'] = bodies[int(doc_id)]
                qas_dict[doc_id]['qas'] = dict()
                qas_dict[doc_id]['qas'][qa_id] = qa_info
            else:
                qas_dict[doc_id]['qas'][qa_id] = qa_info
            previous_doc_id = doc_ids[idx]

        return qas_dict
    
    def _compute_f1(self, a_gold, a_pred): # with word-overlap
        gold_toks = nltk.word_tokenize(a_gold)
        pred_toks = nltk.word_tokenize(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def _compare_pred_gold(self, qas_dict, use_bertscr=True):
        doc_f1_list = []
        for doc_id in qas_dict:
            for qa_id in qas_dict[doc_id]['qas']:
                qa_info = qas_dict[doc_id]['qas'][qa_id]
                if qa_info["reply_scr"] > self.qa_threshold:
                    cand =  qa_info["reply_ans"]
                    ref  =  qa_info['gold_ans']

                    if use_bertscr:
                        _, _, bert_f1 = self.bertscorer.score( [cand], [ref] )
                        bert_f1 = bert_f1.item()
                    else:
                        bert_f1 = self._compute_f1(cand, ref)

                    qas_dict[doc_id]['qas'][qa_id]["bert_f1"] = bert_f1
                    doc_f1_list.append(qa_info["bert_f1"])
                else:
                    qa_info["bert_f1"] = None
            
            if len(doc_f1_list) == 0:
                qas_dict[doc_id]["doc_f1"] = 0
            else:
                qas_dict[doc_id]["doc_f1"] = np.mean(doc_f1_list)
            doc_f1_list = []
        
        f1_list = []
        for doc_id in qas_dict:
            f1_list.append(qas_dict[doc_id]["doc_f1"] )

        return qas_dict, f1_list

    def prevent_no_question_generate(self, summaries, doc_ids, f1_list):
        expect_docs = len(summaries)
        docs_with_score = list(set(doc_ids))

        doc_with_f1 = list(set(doc_ids))
        doc_with_f1.sort()

        cnt = 0
        true_f1_list = []
        for i in range(expect_docs):
            if str(i) not in docs_with_score:
                true_f1_list.append(0)
                cnt += 1
            else:
                target = i - cnt
                true_f1_list.append(f1_list[target])
        return true_f1_list
        
    
    def compute_score(self, bodies, summaries, aggregate=False, show_qas_dict=False, use_bertscr=False):
        #generate questions from summaries
        #print("Generating questions...")

        ts = time.time()
        doc_ids, questions, gold_answers = self._generate_questions(summaries)
        te = time.time()
        #print("time spent generate questions:", te-ts)
        print(doc_ids)

        #print("Getting answers...")
        #run qa system
        ts = time.time()
        gold_answers_dict, squad_format = self._convert_to_squad_format(gold_answers, questions, doc_ids, bodies)
        pred_dict = self._answer_questions_by_context(squad_format)

        te = time.time()
        #print("time spent answering questions:", te-ts)
        
        qas_dict = self._readable_qas_dict(doc_ids, questions, gold_answers, pred_dict, bodies)
        qas_dict, f1_list = self._compare_pred_gold(qas_dict, use_bertscr=use_bertscr)
        f1_list = self.prevent_no_question_generate(summaries, doc_ids, f1_list)
        
        if show_qas_dict:
            for doc_id in qas_dict:
                #print("context:", qas_dict[doc_id]['context'])
                print("doc_f1:", qas_dict[doc_id]['doc_f1'])
                qas = qas_dict[doc_id]['qas']
                for q_id in qas:
                    print(q_id)
                    print("qst:", qas[q_id]["question"])
                    print("g_a:", qas[q_id]["gold_ans"])
                    print("r_a:", qas[q_id]["reply_ans"])
                    print("scr:", qas[q_id]["reply_scr"])
                    print("bert_f1:",  qas[q_id]["bert_f1"] )

        if aggregate:
            return np.mean(f1_list)
        
        return f1_list
    
    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        if extra is None: # sample_score
            scores = self.compute_score( bodies, summaries, aggregate=False, show_qas_dict=False, use_bertscr=False)
            return scores, "no need to cal faithfulness for argmax"
        else: # argmax_scoreZ
            scores = [0] * len(summaries)
            return scores, None


if __name__ == "__main__":
    scorer = FEQA(device='cuda')

    documents = [
                "The world's oldest person has died a \
                few weeks after celebrating her 117th birthday.  \
                Born on March 5, 1898, the greatgrandmother had lived through two world \
                wars, the invention of the television and the \
                first successful powered aeroplane.", 
                "The world's oldest person has died a \
                few weeks after celebrating her 117th birthday.  \
                Born on March 5, 1898, the greatgrandmother had lived through two world \
                wars, the invention of the television and the \
                first successful powered aeroplane."]
    summaries = [
                "The world's oldest person died in 1898",
                "The world's oldest person died after her 117th birthday"]

    ts = time.time()
    score, _= scorer.score(summaries, documents)
    te = time.time()
    print("total time spend {:.2f}".format(te-ts))
    print("final score:", sum(score) / len(score) )