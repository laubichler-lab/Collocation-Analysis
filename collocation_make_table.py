#!/usr/bin/env python3

"""
This program finds collocations in a corpus of text. It can find 
the collocations of keywords you enter manually.
"""

#Import packages
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
from nltk.corpus import stopwords
import pandas as pd




#Import clean data in form of csv, get list of words and remove unfriendly characters
tbl = pd.read_csv('/Users/codyotoole/Desktop/Planetary:OneHealth/Health/text/oh_2020_clean.csv')
text = list(tbl['word'])
text = '\n'.join(text)
data = "".join(i for i in text if ord(i) < 128) 




#tokenize data
tokens = word_tokenize(data)




#empty lists used in function
freq = []
score = []
collocate = []
r = []




def get_keyword_collocations(tokens, keyword, windowsize=10, numresults=35):
    '''This function uses the Natural Language Toolkit to find collocations
    for a specific keyword in a corpus. It takes as an argument a string that
    contains the corpus you want to find collocations from. It prints the top
    collocations it finds for each keyword.
    '''

    # initialize the bigram association measures object to score each collocation
    bigram_measures = BigramAssocMeasures()
    # initialize the bigram collocation finder object to find and rank collocations
    finder = BigramCollocationFinder.from_words(tokens, window_size=windowsize)
    # initialize a function that will narrow down collocates that don't contain the keyword
    keyword_filter = lambda *w: keyword not in w
    # apply a series of filters to narrow down the collocation results
    ignored_words = stopwords.words('english')
    finder.apply_word_filter(lambda w: len(w) < 2 or w.lower() in ignored_words)
    finder.apply_freq_filter(1)
    finder.apply_ngram_filter(keyword_filter)
    # calculate the top results by T-score
    # list of all possible measures: .raw_freq, .pmi, .likelihood_ratio, .chi_sq, .phi_sq, .fisher, .student_t, .mi_like, .poisson_stirling, .jaccard, .dice
    results = finder.score_ngrams(bigram_measures.student_t)
    results = results[:numresults]
    
    t = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
    for p in range(0,len(results)):    
        for n in range(0,len(t)):
            if t[n][0] == results[p][0]:
                freq.append(t[n][1])
    # print the results
    for n in range(0,len(results)):
        r.append(results[n][0])
    print("Top collocations for ", str(keyword), ":")
    print('total occurences of'+' '+keyword+':'+' ',tokens.count(keyword))
    for n in range(0,len(results)):
        score.append(results[n][1])
   
    for k,v in r:
        collocations = ''
        if k != keyword:
                collocations = k
        else:
                collocations = v
        collocate.append(collocations)
        
    


# Replace this with a list of keywords you want to find collocations for
words_of_interest = ["public"]




# Get the top collocations for each keyword in the list above
for word in words_of_interest:
    get_keyword_collocations(tokens, word)
    



#make data frame and then save that frame   
df = pd.DataFrame(
    {'collocate': collocate,
     'frequency': freq,
     'score': score})

df.to_csv('/Users/codyotoole/Desktop/oh_public_2020.csv')
    
