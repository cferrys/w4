import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the training and test datasets
train_file = "BBC News Train.csv"
test_file = "BBC News Test.csv"

# Read the data
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# Inspect the datasets
print("Train Dataset Preview:\n", df_train.head(), "\n")
print("Test Dataset Preview:\n", df_test.head(), "\n")
print("Train Dataset Info:\n", df_train.info(), "\n")
print("Missing values in Train Dataset:\n", df_train.isnull().sum(), "\n")

# Preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Apply preprocessing
df_train['cleaned_text'] = df_train['Text'].apply(preprocess_text)
df_test['cleaned_text'] = df_test['Text'].apply(preprocess_text)

# Visualize category distribution in training set
plt.figure(figsize=(8, 6))
df_train['Category'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Category Distribution in Training Set')
plt.xlabel('Category')
plt.ylabel('Number of Articles')
plt.show()

# Generate a word cloud for training set
text = ' '.join(df_train['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of News Articles (Training Set)')
plt.show()

# Extract features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for computational efficiency

# Transform the cleaned text into numerical data
X_train = vectorizer.fit_transform(df_train['cleaned_text'])
X_test = vectorizer.transform(df_test['cleaned_text'])

print("TF-IDF Train Matrix Shape:", X_train.shape)
print("TF-IDF Test Matrix Shape:", X_test.shape)
