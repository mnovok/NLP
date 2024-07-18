from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open('recenzije.txt', 'r') as file:
    reviews = file.readlines()

with open('ocjene.txt', 'r') as file:
    scores = file.readlines()
scores = [float(score.strip()) for score in scores]    

# Jednojedinično kodiranje recenzija
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews).toarray()

# Pretvaranje ocjena u numpy array
Y = np.array(scores)

# Treniranje linearne regresije
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)

# Predviđanje na temelju treniranog modela 
predictions = linear_regressor.predict(X)

print("Predikcije: ", predictions)
print("Stvarne ocjene: ", Y)

user_review = input("Unesite recenziju: ")

# Jednojedinično kodiranje korisničke recenzije
user_review_vector = vectorizer.transform([user_review]).toarray()
user_prediction = linear_regressor.predict(user_review_vector)

print(f"Recenzija: {user_review}")
print(f"Predviđena ocjena: {user_prediction[0]}")

print("\nBroj recenzija:", len(reviews))
print("Broj ocjena:", len(scores))
print("Ocjene:", scores)

# for review in reviews:
#     print(f"Recenzija: {review.strip()}")
#     print(f"Broj riječi u recenziji: {len(review.strip().split())}")

vocabulary = vectorizer.get_feature_names_out()
print("Rječnik jedinstvenih riječi:", list(vocabulary))
print("Duljina rječnika jedinstvenih riječi:", len(vocabulary))

n = len(vocabulary)
m = len(reviews)
nizovi = [[0 for x in range(n)] for x in range(m)]
print("Prazni nizovi za one-hot encoding:", nizovi)

X_one_hot = X.tolist()
print("Konačni nizovi za one-hot encoding:", X_one_hot)

## Korištenje logističke regresije

print("Rezultati dobiveni logističkom regresijom.\n")
Y = np.array(scores).astype(int)
logistic_regressor = LogisticRegression(max_iter=200, class_weight='balanced')
logistic_regressor.fit(X, Y)

user_review_vector = vectorizer.transform([user_review]).toarray()
user_prediction = logistic_regressor.predict(user_review_vector)

print(f"\nRecenzija: {user_review}")
print(f"Predviđena ocjena: {user_prediction[0]}")
