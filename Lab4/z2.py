from nltk import ngrams
import nltk

sentence = input("Unesite neku recenicu:\n")
n = int(input("Na koliko dijelova zelite podijeliti recenicu?\n"))

n_grams = ngrams(sequence=nltk.word_tokenize(sentence), n=n)

for ngrams in n_grams:
    print(ngrams)
