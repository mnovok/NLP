from googletrans import Translator
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def analyze_review(review):
    tokens = nltk.word_tokenize(review)
    pos_tags = nltk.pos_tag(tokens)

    sentiment_positive = 0.0
    sentiment_negative = 0.0
    
    for word_tag_pair in pos_tags:

        word = word_tag_pair[0]
        pos_tag = word_tag_pair[1]

        if pos_tag.startswith('J'):
            pos_tag =  wn.ADJ
        elif pos_tag.startswith('R'):
            pos_tag =  wn.ADV    
        elif pos_tag.startswith('N'):
            pos_tag =  wn.NOUN
        else:
            continue
        
        wordSynsets = wn.synsets(word, pos=pos_tag)
        if not wordSynsets:
            continue  
        
        chosenSynset = wordSynsets[0]
        sentiWordNet = swn.senti_synset(chosenSynset.name())
        
        sentiment_positive += sentiWordNet.pos_score() 
        sentiment_negative += sentiWordNet.neg_score()

    return sentiment_positive, sentiment_negative


reviews_file = 'recenzije.txt'
with open(reviews_file, 'r', encoding='utf-8') as file:
    reviews = file.readlines()

translator = Translator()
translated_reviews = [translator.translate(review, src='hr', dest='en').text for review in reviews]

for index, review in enumerate(translated_reviews):
    pos_score, neg_score = analyze_review(review)
    
    if pos_score > neg_score:
        print(f"Review {index + 1} is positive.")
    elif pos_score < neg_score:
        print(f"Review {index + 1} is negative.")
    else:
        print(f"Review {index + 1} is neutral.")
