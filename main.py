import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tkinter import *
from tkinter import filedialog

# Sample text data
texts = ["This is a positive sentence.", "This is a negative sentence.", "Another positive example."]

# Tokenization and text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
corpus = []
for text in texts:
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    corpus.append(cleaned_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(corpus)

# Perform K-Means clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(X)
cluster_labels = kmeans.labels_

# Add cluster labels to the original text data
clustered_texts = pd.DataFrame({'Text': texts, 'Cluster': cluster_labels})

# Create a Word Cloud for each cluster
def create_wordcloud(texts, cluster):
    words = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Cluster {cluster} Word Cloud')
    plt.axis('off')
    plt.show()

for cluster in range(num_clusters):
    cluster_texts = clustered_texts[clustered_texts['Cluster'] == cluster]['Text'].values
    create_wordcloud(cluster_texts, cluster)

# Deep Learning Model using Keras and TensorFlow
X_train = X
y_train = cluster_labels

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=2)

# GUI for Text Classification
def classify_text():
    text_to_classify = entry.get()
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text_to_classify.lower())
    words = cleaned_text.split()
    words = [word for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    vectorized_text = vectorizer.transform([cleaned_text])
    cluster = model.predict(vectorized_text)
    cluster_label = 0 if cluster < 0.5 else 1
    result_label.config(text=f'Predicted Cluster: {cluster_label}')

# Create a simple GUI for text classification
root = Tk()
root.title("Text Classification")

frame = Frame(root)
frame.pack(pady=10)

label = Label(frame, text="Enter a sentence:")
label.pack(side=LEFT)

entry = Entry(frame)
entry.pack(side=LEFT)

classify_button = Button(frame, text="Classify", command=classify_text)
classify_button.pack(side=LEFT)

result_label = Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
