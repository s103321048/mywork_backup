import pickle
import time
from eval_FEQA import FEQA
scorer = FEQA(device='cuda')

# load data
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