from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import os

# Define topics and files
topics = {
    'plants': ['sunflower', 'poppy', 'daisy', 'daffodil'],
    'animals': ['lion', 'elephant', 'tiger', 'giraffe'],
    'astronomy': ['moon', 'sun', 'stars', 'planets']
}

# Function to read text files
def read_files(topic, filenames):
    texts = []
    for filename in filenames:
        file_path = os.path.join(topic, f"{filename}.txt")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
    return texts

# Function for k-means clustering training and testing
def kmeans_clustering(topics, files_per_topic_train=3, files_per_topic_test=1):
    train_texts = []
    test_texts = []
    true_labels = []
    
    # Load texts for training
    for topic, files in topics.items():
        train_texts += read_files(topic, files[:files_per_topic_train])
        test_texts += read_files(topic, files[files_per_topic_train:files_per_topic_train + files_per_topic_test])
        true_labels += [topic] * files_per_topic_test

    print(f"Testiraju se datoteke: {test_texts}")    
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # K-means clustering
    k = len(topics)  # Number of clusters equal to number of topics
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=500, n_init=20, tol=1e-4)
    kmeans.fit(X_train)
    
    # Predict clusters for test texts
    predicted_labels = kmeans.predict(X_test)
    
    # Map predicted labels to topic names
    label_mapping = {0: 'plants', 1: 'animals', 2: 'astronomy'}
    mapped_predicted_labels = [label_mapping[label] for label in predicted_labels]
    
    # Accuracy evaluation
    accuracy = accuracy_score(true_labels, mapped_predicted_labels)
    print(f"Accuracy: {accuracy:.2f}")
    
    return true_labels, mapped_predicted_labels

# Call the clustering function with training on 3 documents per topic and testing on 1 document per topic
true_labels, predicted_labels = kmeans_clustering(topics, files_per_topic_train=3, files_per_topic_test=1)

# Print true and predicted labels
print("\nTrue Labels:", true_labels)
print("Predicted Labels:", predicted_labels)
