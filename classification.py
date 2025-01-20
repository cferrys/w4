# classification.py

# === SECTION 1: Import Libraries ===
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# === SECTION 2: Load and Preprocess Data ===
# File paths
train_file = "BBC News Train.csv"
test_file = "BBC News Test.csv"

# Load datasets
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# Inspect datasets
print("Train Dataset Preview:\n", df_train.head(), "\n")
print("Test Dataset Preview:\n", df_test.head(), "\n")

# Preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Apply preprocessing
df_train['cleaned_text'] = df_train['Text'].apply(preprocess_text)
df_test['cleaned_text'] = df_test['Text'].apply(preprocess_text)

# Save cleaned training data
df_train[['Text', 'cleaned_text', 'Category']].to_csv("cleaned_train_data.csv", index=False)
print("Cleaned training data saved to cleaned_train_data.csv")

# Save cleaned test data
if 'Category' in df_test.columns:
    # Include 'Category' if it exists in the test dataset
    df_test[['Text', 'cleaned_text', 'Category']].to_csv("cleaned_test_data.csv", index=False)
else:
    # Exclude 'Category' if it does not exist
    df_test[['Text', 'cleaned_text']].to_csv("cleaned_test_data.csv", index=False)
print("Cleaned test data saved to cleaned_test_data.csv")

# === SECTION 3: Exploratory Data Analysis (EDA) ===
# Visualize category distribution
plt.figure(figsize=(8, 6))
df_train['Category'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Category Distribution in Training Set')
plt.xlabel('Category')
plt.ylabel('Number of Articles')
plt.show()

# Generate a word cloud
text = ' '.join(df_train['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of News Articles (Training Set)')
plt.show()

# === SECTION 4: Feature Extraction with TF-IDF ===
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(df_train['cleaned_text'])
X_test = vectorizer.transform(df_test['cleaned_text'])

print("TF-IDF Train Matrix Shape:", X_train.shape)
print("TF-IDF Test Matrix Shape:", X_test.shape)

# === SECTION 5: Unsupervised Learning with NMF ===
# Number of unique categories
num_categories = len(df_train['Category'].unique())

# Train NMF model
nmf_model = NMF(n_components=num_categories, random_state=42)
W_train = nmf_model.fit_transform(X_train)
W_test = nmf_model.transform(X_test)

# Predict categories
predicted_categories = W_test.argmax(axis=1)
predicted_labels = [df_train['Category'].unique()[i] for i in predicted_categories]

# Evaluate NMF model
if 'Category' in df_test.columns:
    true_labels = df_test['Category']
    nmf_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"NMF Model Accuracy: {nmf_accuracy}")

    # Confusion Matrix for NMF
    nmf_cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix for NMF:\n", nmf_cm)

    # Visualize Confusion Matrix for NMF
    plt.figure(figsize=(8, 6))
    sns.heatmap(nmf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=df_train['Category'].unique(), yticklabels=df_train['Category'].unique())
    plt.title('Confusion Matrix for NMF')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Save NMF predictions
nmf_output = df_test.copy()
nmf_output['Predicted_Category_NMF'] = predicted_labels
nmf_output.to_csv("nmf_predictions.csv", index=False)
print("NMF predictions saved to nmf_predictions.csv")

# === SECTION 6: Supervised Learning with Logistic Regression ===
# Train Logistic Regression model
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, df_train['Category'])

# Predict on test set
supervised_predictions = clf.predict(X_test)

# Evaluate Logistic Regression model
if 'Category' in df_test.columns:
    supervised_accuracy = accuracy_score(df_test['Category'], supervised_predictions)
    print(f"Logistic Regression Model Accuracy: {supervised_accuracy}")

    # Confusion Matrix for Logistic Regression
    supervised_cm = confusion_matrix(df_test['Category'], supervised_predictions)
    print("Confusion Matrix for Logistic Regression:\n", supervised_cm)

    # Visualize Confusion Matrix for Logistic Regression
    plt.figure(figsize=(8, 6))
    sns.heatmap(supervised_cm, annot=True, fmt='d', cmap='Greens', xticklabels=df_train['Category'].unique(), yticklabels=df_train['Category'].unique())
    plt.title('Confusion Matrix for Logistic Regression')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Save Logistic Regression predictions
supervised_output = df_test.copy()
supervised_output['Predicted_Category_LogReg'] = supervised_predictions
supervised_output.to_csv("logreg_predictions.csv", index=False)
print("Logistic Regression predictions saved to logreg_predictions.csv")
