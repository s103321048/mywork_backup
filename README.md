# Boosting Factual Consistency and High Coverage in Unsupervised Abstractive Summarization

## Dependencies
---
- Python 3.6+
- Install all Python packages: `pip install -r requirements.txt`

Leverage with the pre-trained models from [Summary Loop](https://github.com/CannyLab/summary_loop/releases/tag/v0.1), download the following files and place them under _models_ directory. Here are the models needed to run the `train_summary_loop.py`:
- `bert_coverage.bin`: A bert-base-uncased finetuned model on the task of Coverage for the news domain,
- `fluency_news_bs32.bin`: A GPT2 (base) model finetuned on a large corpus of news articles, used as the Fluency model,
- `gpt2_copier23.bin`: A GPT2 (base) model that can be used as an initial point for the Summarizer model.

Also leverage with the qg_model from [FEQA](https://github.com/esdurmus/feqa) and qa_model from [deepset/minilm](https://huggingface.co/deepset/minilm-uncased-squad2):
- Download [checkpoints](https://drive.google.com/drive/folders/1GrnfJxaK35O2IEevv4VbiwYSwxBQVI2X) folder and place it under _bart_qg_ directory.
- No need to install qa_model, it will be automaticly download.

## Data Prepare
---
Follow the instructions [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail) to download CNNDM dataset under _data_ directory. Recommand follow Option1. (See discussion [here](https://github.com/abisee/cnn-dailymail/issues/9) about why we do not provide it ourselves). And see to [create a dataset](https://github.com/CannyLab/summary_loop/blob/master/Dataset%20SQLite3%20Example.ipynb) that will be capable with Summary Loop training script.
1. `cd data`
2. `git clone https://github.com/abisee/cnn-dailymail.git`
3. download CNN_STORIES_TOKENIZED, DM_STORIES_TOKENIZED from [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail) and unzip it
4. `python3 make_datafiles.py`
5. `test_dataset.db` will be create 

## Training Procedure
