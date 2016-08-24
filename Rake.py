import nltk
import re
import operator

debug = False
test = True


def is_number(s):
    try:
        float(s) if '.' is s else int(s)
        return True
    except ValueError:
        return False


def init_stopwords(custom_path=None):
    if not custom_path:
        return nltk.corpus.stopwords.words('english')
    else:
        stop_words = []
        for line in open(custom_path):
            if line.strip()[0:1] != '#':
                for word in line.split():
                    stop_words.append(word)
        nltk_sw = nltk.corpus.stopwords.words('english')
        for word in nltk_sw:
            stop_words.append(word)
        return stop_words


def seperate_words(text, min_word_return_size):
    sentence_re = r'''
    (?:(?:[A-Z])(?:.[A-Z])+.?)
    |(?:\w+(?:-\w+)*)
    |(?:\$?\d+(?:.\d+)?%?)
    |(?:...|)(?:[][.,;"\'?():-_`])
    '''
    return nltk.regexp_tokenize(text, sentence_re)


def split_sentences(text):
    sents =  nltk.sent_tokenize('text')
    print sents
    return sents


def build_stopword_regex(custom_path=None):
    sws = init_stopwords(custom_path)
    print sws
    return sws

def generate_candidate_keywords(sentence_list, stopword_pattern):
    phrase_list = []
    for s in sentence_list:
        phrases = [w for w in s if w not in stopword_pattern]
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != '':
                phrase_list.append(phrase)
    print phrase_list
    return phrase_list

def calculate_word_scores(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = seperate_words(phrase, 0)
        word_list_len = len(word_list)
        word_list_degree = word_list_len - 1
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)
    return word_score


def gen_candidate_keyword_scores(phraseList, word_score):
    keyword_candidates = {}
    for phrase in phraseList:
        keyword_candidates.setdefault(phrase, 0)
        word_list = seperate_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates


class Rake():
    # def __init__(self, stop_words_path):
    def __init__(self):
        # if not stop_words_path:
        # self.stop_words_path = stop_words_path
        self.__stop_words_pattern = init_stopwords()

    def run(self, text):
        sentence_list = split_sentences(text)

        phrase_list = generate_candidate_keywords(sentence_list, self.__stop_words_pattern)

        word_scores = calculate_word_scores(phrase_list)

        keyword_candidates = gen_candidate_keyword_scores(phrase_list, word_scores)

        sorted_keywords = sorted(keyword_candidates.iteritems(),
                                 key=operator.itemgetter(1), reverse=True)
        return sorted_keywords

if test:
    text = '''
    Natural language understanding (NLU) is a subtopic of natural
    language processing in artificial intelligence that deals with
    machine reading comprehension. NLU is considered an AI-hard problem.

    The process of disassembling and parsing input is more complex
    than the reverse process of assembling output in natural language
    generation because of the occurrence of unknown and unexpected
    features in the input and the need to determine the appropriate
    syntactic and semantic schemes to apply to it, factors which
    are pre-determined when outputting language.

    There is considerable commercial interest in the field
    because of its application to news-gathering, text
    categorization, voice-activation, archiving, and
    large-scale content-analysis.
    '''

    sentence_list = split_sentences(text)
    stop_words = init_stopwords()
    phrase_list = generate_candidate_keywords(sentence_list, stop_words)
    word_scores = calculate_word_scores(phrase_list)
    keyword_candidates = gen_candidate_keyword_scores(phrase_list, word_scores)

    if debug:
        print keyword_candidates
        sortedKeywords = sorted(keyword_candidates.iteritems(),
                                key=operator.itemgetter(1), reverse=True)
        print sortedKeywords
        total_keywords = len(sortedKeywords)
        print total_keywords
        print sortedKeywords[0:(total_keywords/3)]

    rake = Rake()
    keywords = rake.run(text)
    print keywords