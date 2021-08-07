# Boosting Factual Consistency and High Coverage in Unsupervised Abstractive Summarization

## Dependencies
---
- Python 3.6+
- Install all Python packages: `pip install -r requirements.txt`

Leverage with the pre-trained models from [Summary Loop](https://github.com/CannyLab/summary_loop/releases/tag/v0.1), download the following files and place them under _models_ directory. Here are the models needed to run the `train_summarizer.py`:
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

Otherwise, you can modify the scripts' data loading (`Dataloader`) and collate function (`collate_fn`) to bring in your own data.

## Training Procedure
Once all the pretraining models and data are ready, train a Summarizer can be done using `train_summarizer.py`:
```
python3 train_summarizer.py --dataset_file {path/to/test_dataset.db} --root_folder {path/to/mywork_backup} --experiment {experiment_name}
```

## Scorer Models (Optional)
The Factual Consistency, Coverage, Fluency models and Brecity can be used separatelt for analysis, evaluation, etc. They are respectively in `model_faith.py`, `model_coverage.py`, `model_generator.py`, `model_guardrails.py`, each model is implemented as a class with a `score(document, summary)` function. 

- Build your own Summarizer & Fluency Scorer

    You can used `utils/train_generator.py` to build your own Summarizer & Fluency model. 
    ```
    python3 train_generator.py --dataset_file {path/to/test_dataset.db} --task {cgen/copy/lm} --max_output_length {23} --experiment {experiment_name}
    ```
    - `cgen` and `copy` is used to create Summarizer.
    - `lm`: is used to create Fluency Scorer.

- Build your own Coverage Scorer

    You can use `utils/pretrain_bert.py` to fine-tune BERT model to your target domain, in our example, news domain.
    ```
    python3 pretrain_bert.py --dataset_file {path/to/test_dataset.db}
    ```
    And used `utils/pretrain_coverage.py` to build Coverage Scorer.
    ```
    python3 pretrain_coverage.py --dataset_file {path/to/test_dataset.db} --experiment {experiment_name}
    ```