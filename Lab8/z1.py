import os
import sentencepiece as spm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

topics = {
    'plants': ['sunflower', 'poppy', 'daisy', 'daffodil'],
    'animals': ['lion', 'elephant', 'tiger', 'giraffe'],
    'astronomy': ['moon', 'sun', 'stars', 'planets']
}

def read_files(topic, filenames):
    texts = []
    for filename in filenames:
        file_path = os.path.join(topic, f"{filename}.txt")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
    return texts

def train_subword_model(texts, model_prefix='subword_model', vocab_size=200):
    with open('corpus.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    spm.SentencePieceTrainer.train(f'--input=corpus.txt --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=unigram')

def tokenize_with_subwords(texts, model_prefix='subword_model'):
    sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
    tokenized_texts = [' '.join(sp.encode(text, out_type=str)) for text in texts]
    return tokenized_texts

def kmeans_clustering(topics, files_per_topic_train=3, files_per_topic_test=1):
    train_texts = []
    test_texts = []
    true_labels = []
    
    for topic, files in topics.items():
        train_texts += read_files(topic, files[:files_per_topic_train])
        test_texts += read_files(topic, files[files_per_topic_train:files_per_topic_train + files_per_topic_test])
        true_labels += [topic] * files_per_topic_test

    # Train subword tokenization model
    train_subword_model(train_texts + test_texts)
    
    # Tokenize texts using subword tokenization
    train_texts_tokenized = tokenize_with_subwords(train_texts)
    test_texts_tokenized = tokenize_with_subwords(test_texts)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=3, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts_tokenized)
    X_test = vectorizer.transform(test_texts_tokenized)
    
    # K-means clustering
    k = len(topics)  # Number of clusters equal to number of topics
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=1000, n_init=50, tol=1e-4)
    kmeans.fit(X_train)
    
    predicted_labels = kmeans.predict(X_test)

    # Map cluster labels to topic names
    label_mapping = {i: topic for i, topic in enumerate(topics.keys())}
    mapped_predicted_labels = [label_mapping[label] for label in predicted_labels]
    
    accuracy = accuracy_score(true_labels, mapped_predicted_labels)
    print(f"Accuracy: {accuracy:.2f}")
    
    return true_labels, mapped_predicted_labels

true_labels, predicted_labels = kmeans_clustering(topics, files_per_topic_train=3, files_per_topic_test=1)

print("\nTrue Labels:", true_labels)
print("Predicted Labels:", predicted_labels)
