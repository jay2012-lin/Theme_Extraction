#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import unicodedata

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

text2 = '''
You’ve probably heard about Pokémon Go, a mobile app that exploded onto the scene in July 2016. Simply explained, to play the game you download the app, peer into your phone and chase imaginary wild Pokémon, flinging Poké Balls at them, and visit PokéStops and gyms to reload and battle, as indicated by your GPS system. We all have fads from our past. I’ll date myself by listing the Rubik’s Cube and a serious addiction to Pac-Man.

This fad is interesting, creative and, from what I understand, addictive. It is also a great way to think about alternatives.

Investors are looking at the real world and choosing to chase something imaginary: growth. We have noted for some time that U.S. potential (or “trend”) growth is falling. Second quarter GDP growth was a paltry 1.2%, only about half of what was expected.1,2 Consumption is an area of the U.S. economy that looks strong, but this was offset by a third consecutive quarter of contracting business investment and a drop in government spending. The fact that U.S. growth is driven primarily by consumption rather than broad-based growth likely makes our economy more vulnerable to interest rate hikes or other exogenous shocks. More importantly, corporate profits are tied to GDP, and corporate profits have been stagnant since 2012. Revenue of S&P 500 companies has fallen five quarters in a row. While recent economic data has “surprised to the upside,” these positive surprises have come mostly from the consumer sector and the housing sector. Equity investors looking at stock indices appear to be chasing economic growth or revenue growth, but both are elusive.

The Pokémon Go craze is a case in point. Shares of the Japanese company Nintendo doubled between July 8 and July 19, adding almost $20 billion to the company’s market capital. Yet many observers noted that Nintendo does not directly own Pokémon Go, and only indirectly benefits from a share of ownership in the company that manages the brand.3 The company may benefit from the popularity of a product, but does the reward of a 100% increase in market cap align with the fact that there was no strategy or value added by the company itself? Perhaps not, as the subsequent decline in Nintendo’s stock price suggests.

It may seem like an extreme example of a unique case, but multiply that by hundreds of companies engaged in corporate buybacks or hitting earnings estimates through cost cutting instead of value harvested from a corporate strategy, and suddenly a broad pattern emerges that reflects the challenges companies and, in turn, investors may face in their quest for future growth.

This summer has been marked by the crosscurrents of geopolitics, changing interest rate expectations and the ebb and flow of economic data releases that cause increased volatility. The strategy of certain alternatives may better tie investor dollars directly to company growth strategies. For example, middle market company revenue has grown 6.7% over the past five quarters. You don’t need an app to see that.
'''

sentence_re = r'''
(?:(?:[A-Z])(?:.[A-Z])+.?)
|(?:\w+(?:-\w+)*)
|(?:\$?\d+(?:.\d+)?%?)
|(?:...|)(?:[][.,;"\'?():-_`])
'''

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()

grammar = r'''
    NBAR:
        {<NN.*|JJ>*<NN.*>}

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}
'''
chunker = nltk.RegexpParser(grammar)
stop_words = nltk.corpus.stopwords.words('english')

decoded_text = unicodedata.normalize('NFKD', text2).encode('ascii', 'ignore')

tokens = nltk.regexp_tokenize(text2, sentence_re)
tagged_tokens = nltk.tag.pos_tag(tokens)

tree = chunker.parse(tagged_tokens)


def leaves(tree):
    '''Finds NP (noun phrase) leaf nodes of a chunk tree'''
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        yield subtree.leaves()


def normalize(word):
    '''Normalizes words to lower case and stems / lemmatizes them.'''
    word = word.lower()
    word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word


def acceptable_word(word):
    '''Checks conditions for acceptable words: length, stopwords'''
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stop_words)
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        term = [normalize(w) for w, t in leaf if acceptable_word(w)]
        yield term


terms = get_terms(tree)
title = nltk.Text(text2)

print title
for term in terms:
    for word in term:
        print word,
    print
