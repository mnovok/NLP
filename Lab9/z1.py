import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define the topics and corresponding file names
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

def preprocess_text(texts):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    processed_texts = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        processed_texts.append(' '.join(filtered_tokens))
    
    return processed_texts

def kmeans_clustering(topics, files_per_topic_train=3, files_per_topic_test=1):
    train_texts = []
    test_texts = []
    true_labels = []
    
    for topic, files in topics.items():
        train_texts += read_files(topic, files[:files_per_topic_train])
        test_texts += read_files(topic, files[files_per_topic_train:files_per_topic_train + files_per_topic_test])
        true_labels += [topic] * files_per_topic_test
    
    # Preprocess texts
    train_texts_processed = preprocess_text(train_texts)
    test_texts_processed = preprocess_text(test_texts)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts_processed)
    X_test = vectorizer.transform(test_texts_processed)
    
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
