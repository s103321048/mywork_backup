import sys
import os
import hashlib
import subprocess
import collections

import json
import tarfile
import io
import pickle as pkl

import sqlite3
from sqlite3 import Error
import time
from tqdm import tqdm

dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]

all_train_urls = "cnn-dailymail/url_lists/all_train.txt"
all_val_urls = "cnn-dailymail/url_lists/all_val.txt"
all_test_urls = "cnn-dailymail/url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
# finished_files_dir = "finished_files"

# These are the number of .story files we expect there to be in cnn_stories_dir
# and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

def read_story_file(text_file):
    with open(text_file, "r") as f:
        # sentences are separated by 2 newlines
        # single newlines might be image captions
        # so will be incomplete sentence
        lines = f.read().split('\n\n')
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def get_art_abs(story_file): # get article abstract sentence (highlight)
    """ return as list of sentences"""
    lines = read_story_file(story_file)

    # Lowercase, truncated trailing spaces, and normalize spaces
    lines = [' '.join(line.strip().split()) for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem
    # in the dataset because many image captions don't end in periods;
    # consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    return article_lines, highlights


def insert_to_db(url_file, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the
       url_file and writes them to a out_file.
    """
    ### creat database
    print(out_file)
    db_file = out_file
    if os.path.isfile(db_file):
        os.remove(db_file)
    conn = sqlite3.connect(db_file,detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # creat table
    sql_create = "CREATE TABLE articles (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, body TEXT);"
    c.execute(sql_create)
    conn.commit()

    print("Making bin file for URLs listed in {}...".format(url_file))
    url_list = [line.strip() for line in open(url_file)]
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s+".story" for s in url_hashes]
    num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    ts = time.time()
    for idx, s in enumerate(tqdm(story_fnames)):
        if idx % 1000 == 0:
            print("Writing story {} of {}; {:.2f} percent done. Time spent: {:.2f}".format(
                idx, num_stories, float(idx)*100.0/float(num_stories), time.time() - ts ))
            if idx != 0:
                conn.commit() # commet sql tasks
        # Look in the tokenized story dirs to find the .story file
        # corresponding to this url
        if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
            story_file = os.path.join(cnn_tokenized_stories_dir, s)
        elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
            story_file = os.path.join(dm_tokenized_stories_dir, s)
        else:
            print("Error: Couldn't find tokenized story file {} in either"
                    " tokenized story directories {} and {}. Was there an"
                    " error during tokenization?".format(
                        s, cnn_tokenized_stories_dir,
                        dm_tokenized_stories_dir))
            # Check again if tokenized stories directories contain correct
            # number of files
            print("Checking that the tokenized stories directories {}"
                    " and {} contain correct number of files...".format(
                        cnn_tokenized_stories_dir, dm_tokenized_stories_dir))
            check_num_stories(cnn_tokenized_stories_dir,
                                num_expected_cnn_stories)
            check_num_stories(dm_tokenized_stories_dir,
                                num_expected_dm_stories)
            raise Exception(
                "Tokenized stories directories {} and {}"
                " contain correct number of files but story"
                " file {} found in neither.".format(
                    cnn_tokenized_stories_dir,
                    dm_tokenized_stories_dir, s)
            )

        # Get the strings to write to .bin file
        article_sents, abstract_sents = get_art_abs(story_file)

        # Write to .db file
        body = " ".join( article_sents ) 
        title = " ".join( abstract_sents ) 
        sql_insert = "INSERT INTO articles (title, body) VALUES (?, ?)"
        c.execute(sql_insert, (title, body))
        
    
        # Write the vocab to file, if applicable
        if makevocab:
            art_tokens = ' '.join(article_sents).split()
            abs_tokens = ' '.join(abstract_sents).split()
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens] # strip
            tokens = [t for t in tokens if t != ""] # remove empty
            vocab_counter.update(tokens)

    conn.commit()
    print("Finished writing file {}\n".format(out_file))

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        #with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"),
        with open("vocab_cnt.pkl",'wb') as vocab_file:
            pkl.dump(vocab_counter, vocab_file)
        print("Finished writing vocab file")

def save_as_pickle(url_file, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the
       url_file and writes them to a out_file.
    """
    news_dict_list = []

    print(out_file)
    db_file = out_file
    if os.path.isfile(db_file):
        os.remove(db_file)

    print("Making bin file for URLs listed in {}...".format(url_file))
    url_list = [line.strip() for line in open(url_file)]
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s+".story" for s in url_hashes]
    num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    ts = time.time()
    for idx, s in enumerate(tqdm(story_fnames)):
        if idx % 1000 == 0:
            print("Writing story {} of {}; {:.2f} percent done. Time spent: {:.2f}".format(
                idx, num_stories, float(idx)*100.0/float(num_stories), time.time() - ts ))

        # Look in the tokenized story dirs to find the .story file
        # corresponding to this url
        if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
            story_file = os.path.join(cnn_tokenized_stories_dir, s)
        elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
            story_file = os.path.join(dm_tokenized_stories_dir, s)
        else:
            print("Error: Couldn't find tokenized story file {} in either"
                    " tokenized story directories {} and {}. Was there an"
                    " error during tokenization?".format(
                        s, cnn_tokenized_stories_dir,
                        dm_tokenized_stories_dir))
            # Check again if tokenized stories directories contain correct
            # number of files
            print("Checking that the tokenized stories directories {}"
                    " and {} contain correct number of files...".format(
                        cnn_tokenized_stories_dir, dm_tokenized_stories_dir))
            check_num_stories(cnn_tokenized_stories_dir,
                                num_expected_cnn_stories)
            check_num_stories(dm_tokenized_stories_dir,
                                num_expected_dm_stories)
            raise Exception(
                "Tokenized stories directories {} and {}"
                " contain correct number of files but story"
                " file {} found in neither.".format(
                    cnn_tokenized_stories_dir,
                    dm_tokenized_stories_dir, s)
            )

        # Get the strings to write to .bin file
        article_sents, abstract_sents = get_art_abs(story_file)

        # Write to .db file
        body = " ".join( article_sents )
        title = " . ".join( abstract_sents )

        news_dict = dict()
        news_dict["content"] = body
        news_dict["summary"] = title
        news_dict_list.append(news_dict)

        # Write the vocab to file, if applicable
        if makevocab:
            art_tokens = ' '.join(article_sents).split()
            abs_tokens = ' '.join(abstract_sents).split()
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens] # strip
            tokens = [t for t in tokens if t != ""] # remove empty
            vocab_counter.update(tokens)

    print("Finished writing file {}\n".format(out_file))
    with open(out_file, "wb") as f:
        pickle.dump(news_dict_list, f)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"),
                  'wb') as vocab_file:
            pickle.dump(vocab_counter, vocab_file)

        print("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory {} contains {} files"
            " but should contain {}".format(
                stories_dir, num_stories, num_expected)
        )

if __name__ == '__main__':
    save_as_pickle(all_test_urls, "test_dataset.pkl")
    insert_to_db(  all_test_urls, "test_dataset.db")

    # save_as_pickle(all_val_urls, "val_dataset.pkl")
    # insert_to_db(  all_val_urls, "val_dataset.db")

    # save_as_pickle(all_train_urls, "train_dataset.pkl")
    # insert_to_db(  all_train_urls, "train_dataset.db")
