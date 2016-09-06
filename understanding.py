# !/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import unicode_literals
from __future__ import division
import unicodedata
import operator
from nltk.corpus import PlaintextCorpusReader
import nltk
import string
import itertools
import gensim
import re
import os
import sys


def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks,
                                                      lambda (word, pos, chunk): chunk != 'O') if key]

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]


def extract_candidate_words(text, good_tags=set(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])):
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates


def score_keyphrases_by_tfidf(texts, candidates='chunks'):
    # extract candidates from each text in texts, either chunks or words
    if candidates == 'chunks':
        boc_texts = [extract_candidate_chunks(text) for text in texts]
    elif candidates == 'words':
        boc_texts = [extract_candidate_words(text) for text in texts]
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    return corpus_tfidf, dictionary


def score_keyphrases_by_textrank(text, n_keywords=0.05):
    from itertools import takewhile, tee, izip
    import networkx

    # tokenize for all words, and extract *candidate* words
    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = extract_candidate_words(text)
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))

    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i + 10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)

    return sorted(keyphrases.iteritems(), key=lambda x: x[1], reverse=True)


def is_punct(word):
    return len(word) == 1 and word in string.punctuation


def is_numeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False


def seperate_words(text, min_word_size):
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/|]')
    words = []
    for single_word in splitter.split(text):
        current_word = normalize_words(single_word)
        if len(current_word) > min_word_size and current_word != '' and not is_numeric(current_word):
            words.append(current_word)
    return words


def build_stopword_pattern(stopwords):
    stop_word_regex_list = []
    for word in stopwords:
        word_regex = r'\b' + word + r'(?![\w-])'
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern


def normalize_words(word):
    stem = nltk.stem.porter.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()

    word = word.lower()
    word = stem.stem_word(word)
    word = lemmatizer.lemmatize(word)

    return word


class RakeKeywordExtractor:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words()
        self.stopword_pattern = build_stopword_pattern(self.stopwords)
        self.top_fraction = 0.5  # consider top third candidate keywords by score
        self._include_scores = False
        self._add_to_corpus = True
        self._load = False
        self._text = ''

        if os.name != 'nt':
            self._my_corpus_root = '/Users/rickyschools/Documents/financial_world_corpus/'
        else:
            self._my_corpus_root = 'C:/Python27/financial_world_corpus/'
        self._fs_root = self._my_corpus_root + 'FS perspectives/'
        self._FS_Corpus = PlaintextCorpusReader(self._my_corpus_root, '.*', encoding='ascii')
        # self._FS_Corpus = SeekableUnicodeStreamReader(self._my_corpus_root, '.*')

    def _generate_candidate_keywords(self, sentences, stop_word_pattern):
        phrase_list = []
        for sentence in sentences:
            tmp = re.sub(stop_word_pattern, '|', sentence.strip())
            phrases = tmp.split('|')
            for phrase in phrases:
                phrase = phrase.strip().lower()
                if phrase != '':
                    phrase_list.append(phrase)
        return phrase_list

    def _calculate_word_scores(self, phrase_list):
        word_freq = {}
        word_degree = {}
        for phrase in phrase_list:
            word_list = seperate_words(phrase, 0)
            word_list_len = len(word_list)
            word_list_degree = word_list_len - 1
            for word in word_list:
                word_freq.setdefault(word, 0)
                word_freq[word] += 1
                word_degree.setdefault(word, 0)
                word_degree[word] += word_list_degree

        for item in word_freq:
            word_degree[item] = word_degree[item] / (word_freq[item] * 1.0)

        word_score = {}
        for item in word_freq:
            word_score.setdefault(item, 0)
            word_score[item] = word_degree[item] / (word_freq[item] * 1.0)
        return word_score

    def _calculate_phrase_scores(self, phrase_list, word_scores):
        keyword_candidates = {}
        for phrase in phrase_list:
            keyword_candidates.setdefault(phrase, 0)
            word_list = seperate_words(phrase, 0)
            candidate_score = 0
            for word in word_list:
                candidate_score += word_scores[word]
            keyword_candidates[phrase] = candidate_score
        return keyword_candidates

    def __load_corpus(self, load, root):
        print('this function would load corpus data'
              'in %s if load is %s' % (root, load))
        # need to check for valid root corpus
        # need to determine how to establish understanding

    def __print_file_ids(self):
        file_ids = self._FS_Corpus.fileids()
        print('Which FS perspective file should I load?\n')
        for f in file_ids:
            if f != '.DS_Store':
                print f
        return raw_input('Your input: >>> '), file_ids

    def __lazy_load_corpus_text(self, count=0):
        file_name, file_ids = self.__print_file_ids()
        file_ids = [unicodedata.normalize('NFKD', f).encode('ascii', 'ignore') for f in file_ids]
        # file_ids = [f.encode('ascii', 'ignore').decode('utf8') for f in file_ids]
        while count < 2:
            if file_name not in file_ids:
                count += 1
                print("I don't know that file. %s tries left." % (2 - count))
                if count == 2:
                    raise BaseException
                    sys.exit()
            else:
                print self._FS_Corpus.raw(file_name)
                sys.exit(1)
                # return self._FS_Corpus.raw(file_name)

    def extract(self, text='', include_scores=False, load=False, load_root=''):
        self._include_scores = include_scores
        if len(text) < 1:
            self._text = self.__lazy_load_corpus_text()
            print('Article: %s\n' % self._text)

        else:
            self._text = text
        self._load = load
        if self._load:
            print('Would load in all understanding data for %s.' % load_root)

        sentences = nltk.sent_tokenize(self._text)
        phrase_list = self._generate_candidate_keywords(sentences, self.stopword_pattern)
        word_scores = self._calculate_word_scores(phrase_list)
        phrase_scores = self._calculate_phrase_scores(phrase_list, word_scores)
        sorted_phrase_scores = sorted(phrase_scores.iteritems(),
                                      key=operator.itemgetter(1), reverse=True)
        n_phrases = len(sorted_phrase_scores)
        if include_scores:
            return sorted_phrase_scores[0:int(n_phrases / self.top_fraction)]
        else:
            return map(lambda x: x[0],
                       sorted_phrase_scores[0:int(n_phrases / self.top_fraction)])

    def __user_folder_input(self, folders):
        user_folder = raw_input('Adding to my corpus.'
                                'Where should I put this?\n\n' % '\n'.join(folders))
        return user_folder

    def __user_file_name_input(self):
        return raw_input('What should we call this file?')

    def __add_input_to_corpus(self, add_to_corpus=False, count=0):
        self._add_to_corpus = add_to_corpus
        folders = [d for d in os.listdir(self._my_corpus_root) if
                   os.path.isdir(os.path.join(self._my_corpus_root, d))]

        if self._add_to_corpus:
            while count < 2:
                user_folder = self.__user_folder_input(folders)
                if user_folder not in folders:
                    print("It looks like that folder doesn't exist. Can you try again?")
                    count += 1
                else:
                    print("Great. I'll save it in the %s folder." % user_folder)
                    break
            user_file_name = self.__user_file_name_input()
        file_name = self._my_corpus_root + user_folder + '/' + user_file_name + '.txt'

        with open(file_name, 'w') as text_file:
            text_file.write(self._text)
        text_file.close()
        print('%s added to the %s corpus.' % (user_file_name, user_folder))


# text_decoded = unicodedata.normalize('NFKD', article_text6).encode('ascii', 'ignore')
# tokens = nltk.wordpunct_tokenize(text_decoded)
# text = nltk.Text(tokens)

rake = RakeKeywordExtractor()
keywords = rake.extract(include_scores=True)
print('\nKey Themes:\n')

for kp in keywords[:10]:
    print(kp)

# key_phrases = score_keyphrases_by_textrank(text_decoded)
# for kp in key_phrases:
#     print kp
