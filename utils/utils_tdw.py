import spacy
import networkx as nx 
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_sm")

def sort_dict_by_value(input_dict):
    return dict(sorted(input_dict.items(), key=lambda k: k[1], reverse=True))

def load_stop_words(stopwords=""):
    stop_words = []
    for word in STOP_WORDS.union(set(stopwords.split(" "))):
        stop_words.append(word)
    return stop_words

def ann_text(text, all_dep=False):
    #no_pronoun_text = remove_pronoun(text)
    #doc = nlp(no_pronoun_text)
    doc = nlp(text)
    dep_list = []
    for sent_idx, sent in enumerate(doc.sents):
        for tok_idx, tok in enumerate(sent):
            dep = find_dep(tok, all_dep)
            if dep is not None:
                dep_list.extend( dep )
    if all_dep is False:
        merge_compound_word(dep_list, doc)
#         print("===== next semt =====")
    return dep_list

def dep_fwfw(tok, dp1, dp2, double_dir=False, reverse=False): # dep_forward_forward
    # A->B->C  => C->A
    #     ex: table for food
    #         dep = (table-->for), (for-->food)
    #             => table<--food
    source = None
    target = None
    for child in tok.children:
        if child.dep_ == dp1:
            for cchild in child.children:
                if cchild.dep_ == dp2:
                    source = tok
                    target = cchild
    return creat_dep(source, target, double_dir, reverse)

def dep_side(tok, dp1, dp2, double_dir=False, reverse=False):
    # A<-B->C => C->A
    #     ex: He is nice.   dep = (is->he), (is->nice) 
    #                           => nice->he
    source = None
    target = None
    for child in tok.children:
        if child.dep_ == dp1:
            target = child
        if child.dep_ == dp2:
            source = child
    return creat_dep(source, target, double_dir, reverse)

def dep_rem(tok, dp, double_dir=False, reverse=False): # dep_remain
    # A->B => A->B
    #     ex: find->it     dep = (find->it)
    #                          => find->it
    source = None
    target = None
    for child in tok.children:
        if child.dep_ == dp:
            source = tok
            target = child
    return creat_dep(source, target, double_dir, reverse)
    
    
def creat_dep(source, target, double_dir=False, reverse=False):
    if source is not None and target is not None:
        if double_dir:
            return [( tok2node(source), tok2node(target) ), 
                    ( tok2node(target), tok2node(source) )]
        else:
            if reverse: # source -> target
                return (tok2node(target), tok2node(source) )
            else:
                return (tok2node(source), tok2node(target) )
    else:
        return None

def tok2node(token): # token => str( lemma=pos )
    if type(token) is str:
        return token
    else:
        tok_lemma = token.lemma_
        if token.lemma_ == "-PRON-":
            tok_lemma = token.text.lower()
        if token.pos_ == "PUNCT":
            return token.text
        else:
            return tok_lemma
    #        return "".join([tok_lemma, '=', token.pos_])

def merge_compound_word(dep_list, doc):
    compound_word_list, all_dep_in_cpw_list = find_cpw(doc)
    
    cpw_dict = dict()
    for cpw in compound_word_list:
        main_cpw = cpw[0]
        part_cpw = cpw[1]
        cpw_dict[part_cpw] = main_cpw
    
    for idx, dep in enumerate(dep_list):
        source = dep[0]
        target = dep[1]
        if source in cpw_dict:
            dep_list[idx] = (cpw_dict[source], target)
        elif target in cpw_dict:
            dep_list[idx] = (source, cpw_dict[target])
    for dep_in_cpw in all_dep_in_cpw_list:
        dep_list.extend(dep_in_cpw)

def find_cpw(doc):
    compound_word_list = []
    all_dep_in_cpw_list = []
    tmp_cpw_word = []
    cpw_head = ""
    for i in doc:
        if i.dep_ == "compound":
            cpw_head = i.head.text
            if cpw_head not in tmp_cpw_word:
                tmp_cpw_word.append(i.text)
        else:
            if i.text == cpw_head:
                tmp_cpw_word.append(cpw_head)
                cpw, cpw_dep_list = dep_in_cpw(tmp_cpw_word)
                compound_word_list.append( (cpw, i.text) )
                all_dep_in_cpw_list.extend(cpw_dep_list)
                tmp_cpw_word = []
    return compound_word_list, all_dep_in_cpw_list

def dep_in_cpw(tmp_cpw_word):
    cpw = " ".join(tmp_cpw_word)
    cpw_dep_list = []            # Robin Kuo <-> Robin
    for t in tmp_cpw_word:       #           <-> Kuo
        cpw_dep_list.append(creat_dep(cpw, t, double_dir=True))
    return cpw, cpw_dep_list

def find_dep(tok, all_dep=False):
    tmp_dep_list = []
    if all_dep:
        tmp_dep_list.append( deps(tok))
    else:
        tmp_dep_list.append( nsubj(tok) )
        tmp_dep_list.append( nsubjp(tok))
        tmp_dep_list.append( same(tok)  )
        tmp_dep_list.append( ppobj(tok) )
        tmp_dep_list.append( agpbj(tok) )
        tmp_dep_list.append( dative(tok))
        tmp_dep_list.append( ppcomp(tok))
        tmp_dep_list.append( advcl(tok) )
        tmp_dep_list.append(npadvmod(tok))
        tmp_dep_list.append( amod(tok)  )
        tmp_dep_list.append( advmod(tok))
        tmp_dep_list.append( dobj(tok)  )
        tmp_dep_list.append( nummod(tok))
        tmp_dep_list.append( xcomp(tok) )
        tmp_dep_list.append( ccomp(tok) )
        tmp_dep_list.append( acl(tok)   )
        tmp_dep_list.append( poss(tok)  )
        tmp_dep_list.append( relcl(tok) )
        tmp_dep_list.append( oprd(tok)  )
        #tmp_dep_list.append(compound(tok))
    return flat_rmNone_list(tmp_dep_list)

def flat_rmNone_list(tmp_dep_list):
    dep_list = []
    for dep in tmp_dep_list:
        if dep is None:
            continue
        elif type(dep) is list:
            dep_list.extend(dep)
        else:
            dep_list.append(dep)
    return dep_list

def deps(tok):
    source = None
    target = None
    dep_list = []
    for child in tok.children:
        source = tok
        target = child
        dep_list.append(creat_dep(source, target))
    return flat_rmNone_list(dep_list)

def nsubj(tok):
    dep = dep_side(tok, 'nsubj', 'acomp') # "I am nice"
    if dep is None:
        dep = dep_side(tok, 'nsubj', 'attr', double_dir=True)  #"Tom is man"
    if dep is None:
        dep = dep_rem(tok, 'nsubj') # "I ran home"
    return dep
def nsubjp(tok):
    return dep_rem(tok, 'nsubjpass') # "dog was found"
def same(tok):
    return dep_rem(tok, 'appos', double_dir=True) # "Sam, the VIP"
def agpbj(tok): # agent+pobj
    return dep_fwfw(tok, 'agent', 'pobj') # "taken by us"
def ppobj(tok): # prep+pobj
    if tok.pos_ in ['NOUN', 'PROPN']: 
        return dep_fwfw(tok, 'prep', 'pobj', reverse=True) # "table of picnic"
    else:                             
        return dep_fwfw(tok, 'prep', 'pobj') # "some of toys"
def dative(tok):
    return dep_rem(tok, 'dative') # "gave me book"
def ppcomp(tok):
    return dep_fwfw(tok, 'prep', 'pcomp') # "play at flying"
def advcl(tok):
    return dep_rem(tok, 'advcl') #"cry when fail"
def npadvmod(tok):
    return dep_rem(tok, 'npadvmod', reverse=False) # "done this morning"
def amod(tok):
    return dep_rem(tok, 'amod', reverse=True) # "poor student"
def advmod(tok):
    return dep_rem(tok, 'advmod', reverse=True) # "less often"
def dobj(tok):
    return dep_rem(tok, 'dobj') # "find it"
def nummod(tok):
    return dep_rem(tok, 'nummod', reverse=True) # "ten books"
def xcomp(tok):
    return dep_rem(tok, "xcomp") # "easy to play"
def ccomp(tok):
    return dep_rem(tok, "ccomp") # "I conside him fool"
def poss(tok):
    return dep_rem(tok, "poss", reverse=True) # "his gun"
def relcl(tok):
    return dep_rem(tok, "relcl", reverse=True) # "girl who likes me"
def oprd(tok):
    return dep_rem(tok, "oprd") # "made public"
def acl(tok):
    return dep_rem(tok, "acl", reverse=True) # "fact that nobody care"
def compound(tok):
    return dep_rem(tok, "compound", double_dir=True)


def mydep_node_dict(text, direct=True, in_edge=True, show_graph=False, all_dep=False, div_wcnt=False):
    dep_list = ann_text(text, all_dep=all_dep)

    if direct:
        G = nx.DiGraph()
    else:
        G = nx.MultiGraph() 
    for i in dep_list: # [('cool=JJ', 'he=NN', 'nsubj')]
        G.add_edge(i[0], i[1]) 
    if show_graph:
        nx.draw_networkx(G, with_label = True, node_color="yellow")
    
    # ======= calculate weight ======= 
    # count how many edge attach to each node
    node_edge_cnt_dict = dict()
    for node in G.nodes:
        if direct==True and in_edge==True:
            node_edge_cnt_dict[node] = len(G.in_edges(node))
        else: # out_edge
            node_edge_cnt_dict[node] = len(G.edges(node))

    node_word_cnt_dict = word_freq_cnt(dep_list)


    weight_node_dict = dict()
    for node in node_edge_cnt_dict.keys():
        if div_wcnt:
            weight_node_dict[node] = node_edge_cnt_dict[node] /node_word_cnt_dict[node]
        else:
            weight_node_dict[node] = node_edge_cnt_dict[node]
    return sort_dict_by_value(weight_node_dict)

def word_freq_cnt(dep_list):
    freq_dict = {}
    for node_s, node_t in dep_list:
        if node_t not in freq_dict:
            freq_dict[node_t] = 1
        else:
            freq_dict[node_t] += 1

        if node_s not in freq_dict:
            freq_dict[node_s] = 1
        else:
            freq_dict[node_s] += 1
    return freq_dict

def mydep_score(text_list, all_dep=False, keystr=False):
    mydep_list = []
    for text in text_list:
        keyword_dict = mydep_dict(text, all_dep=all_dep, keystr=keystr)       
        mydep_list.append( keyword_dict )
    return mydep_list

def mydep_dict(text, all_dep=False, keystr=False):
    node_dict = mydep_node_dict(text, all_dep=all_dep)
    stopw = load_stop_words()
    keyword_dict = dict()

    if keystr:
        for node in node_dict:
            if len(node.split(" ")) == 1:
                kw = node.lower()
                if kw not in stopw:
                    keyword_dict[kw] = node_dict[node]   
    else:
        for node in node_dict:
            kw = node.lower()
            if kw not in stopw:
                keyword_dict[kw] = node_dict[node]    
    return sort_dict_by_value(keyword_dict)