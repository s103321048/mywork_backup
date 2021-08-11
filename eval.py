from model_generator import GeneTransformer
from tqdm import tqdm
import pickle
from utils.eval_rouge import cal_rouge
from utils.eval_FEQA import cal_FEQA

test = 5

# load your model
model_name = "" # "summarizer_{experiment_name}_ckpt.bin"
generator = GeneTransformer(device="cuda") # Initialize the generator
generator.reload("models/{}".format(model_name)) 

# load dataset (pickle)
with open ("data/test_dataset.pkl", "rb") as f:
    news_list = pickle.load(f)


content_list = []
gold_list = []
for news in news_list:
    content_list.append(news['content'])
    gold_list.append(news['summary'])
print(len(content_list), len(gold_list))

# generate summary
cand_list = []
for idx, new in tqdm(enumerate(news_list)):
    if idx == test:
        break

    summary = generator.decode([new['content']], max_output_length=20, beam_size=1, return_scores=True, sample=False)
    if summary == "":
        cand_list.append("None")
    else:
        cand_list.append(summary[0][0])


print( " ==================== " )
print("doc/cand/gold = {}/{}/{}",len(content_list[:test]), len(cand_list), len(gold_list[:test]))
result_FEQA = cal_FEQA(cand_list, content_list[:test])
result_rouge = cal_rouge(gold_list[:test], cand_list)
