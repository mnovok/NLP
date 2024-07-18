from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import os

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

def kmeans_clustering(topics, files_per_topic_train=3, files_per_topic_test=1):
    train_texts = []
    test_texts = []
    true_labels = []
    
    for topic, files in topics.items():
        train_texts += read_files(topic, files[:files_per_topic_train])
        test_texts += read_files(topic, files[files_per_topic_train:files_per_topic_train + files_per_topic_test])
        true_labels += [topic] * files_per_topic_test

    print(f"Testiraju se datoteke: {test_texts}")    
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=3, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # K-means clustering
    k = len(topics)  # Number of clusters equal to number of topics
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=1000, n_init=50, tol=1e-4)
    kmeans.fit(X_train)
    
    predicted_labels = kmeans.predict(X_test)

    label_mapping = {0: 'plants', 1: 'animals', 2: 'astronomy'}
    mapped_predicted_labels = [label_mapping[label] for label in predicted_labels]
    
    accuracy = accuracy_score(true_labels, mapped_predicted_labels)
    print(f"Accuracy: {accuracy:.2f}")
    
    return true_labels, mapped_predicted_labels

true_labels, predicted_labels = kmeans_clustering(topics, files_per_topic_train=3, files_per_topic_test=1)

print("\nTrue Labels:", true_labels)
print("Predicted Labels:", predicted_labels)
