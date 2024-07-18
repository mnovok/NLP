import numpy as np
from collections import defaultdict
import re

def load_data(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def word_count(reviews):
    word_freq = defaultdict(int)
    for review in reviews:
        words = re.findall(r'\b\w+\b', review.lower())
        for word in words:
            word_freq[word] += 1
    return word_freq    

def word_prob(word, word_count, total_words, vocab_size):
    return (word_count[word] + 1) / (total_words + vocab_size)

def review_prob(review, class_word_probs, total_words):
    words = re.findall(r'\b\w+\b', review.lower())
    log_prob = np.log(1)
    
    for word in words:
        if word in class_word_probs:
            log_prob += np.log(class_word_probs[word])
        else:
            log_prob += np.log(1 / total_words)  
    
    return log_prob

def classify_reviews(pos_probs, neg_probs):
    predictions = []
    for pos_prob, neg_prob in zip(pos_probs, neg_probs):
        if pos_prob > neg_prob:
            predictions.append(1)  # Class 1 je pozitivna
        else:
            predictions.append(0)  # Class 0 je negativna
    return predictions
    
train_reviews = load_data('recenzijeTrain.txt')
train_labels = [int(label) for label in load_data('klaseTrain.txt')]

test_reviews = load_data('recenzijeTest.txt')
test_labels = [int(label) for label in load_data('klaseTest.txt')]

pos_reviews = [train_reviews[i] for i in range(len(train_reviews)) if train_labels[i] == 1]
neg_reviews = [train_reviews[i] for i in range(len(train_reviews)) if train_labels[i] == 0]

pos_word_count = word_count(pos_reviews)
neg_word_count = word_count(neg_reviews)

total_pos_words = sum(pos_word_count.values())
total_neg_words = sum(neg_word_count.values())

vocab = set(pos_word_count.keys()).union(set(neg_word_count.keys()))
vocab_size = len(vocab)

print(f'Ukupan broj riječi u pozitivnim recenzijama: {total_pos_words}')
print(f'Ukupan broj riječi u negativnim recenzijama: {total_neg_words}')
print(f'Ukupan broj jedinstvenih riječi (vokabular): {vocab_size}')

pos_word_probs = {word: word_prob(word, pos_word_count, total_pos_words, vocab_size) for word in vocab}
neg_word_probs = {word: word_prob(word, neg_word_count, total_neg_words, vocab_size) for word in vocab}

pos_review_probs = [review_prob(review, pos_word_probs, total_pos_words) for review in test_reviews]
neg_review_probs = [review_prob(review, neg_word_probs, total_neg_words) for review in test_reviews]

predictions = classify_reviews(pos_review_probs, neg_review_probs)

accuracy = np.mean(predictions == test_labels) * 100
print(f"Točnost Bayesova algoritma: {accuracy:.2f}%")

# Debugging output
print("Predictions:", predictions)
print("Test Labels:", test_labels)
