import pickle
import time
import numpy
from tqdm import tqdm
from rouge import Rouge 
def cal_rouge(gold_list, cand_list):
    rouge = Rouge()
    score_list = []

    ts = time.time()
    for gold, cand in tqdm(zip(gold_list, cand_list)):
        scores = rouge.get_scores(gold, cand)
        score_list.append(scores[0])
    te = time.time()
    print("time spend cal_rouge:{:.2f}".format(te-ts))

    result_rouge = dict()
    result_rouge['rouge1'] = []
    result_rouge['rouge2'] = []
    result_rouge['rougeL'] = []
    for s in score_list:
        result_rouge['rouge1'].append(s['rouge-1']['f'])
        result_rouge['rouge2'].append(s['rouge-2']['f'])
        result_rouge['rougeL'].append(s['rouge-l']['f'])

    result_rouge['rouge1'] = numpy.array(result_rouge['rouge1'])
    result_rouge['rouge2'] = numpy.array(result_rouge['rouge2'])
    result_rouge['rougeL'] = numpy.array(result_rouge['rougeL'])

    print("rouge1:", result_rouge['rouge1'].mean() )
    print("rouge2:", result_rouge['rouge2'].mean() )
    print("rougeL:", result_rouge['rougeL'].mean() )
    return result_rouge

if __name__ == "__main__":
    gold_list = ['police killed the gunman']
    cand_list = ['police kill the gunman']

    print("reference:", gold_list[0])
    print("summary:", cand_list[0])
    cal_rouge(gold_list, cand_list)