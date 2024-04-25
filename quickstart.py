# ALL IMPORTS
from apiclient import discovery
from apiclient import errors
from httplib2 import Http
from oauth2client import file, client, tools
import base64
from bs4 import BeautifulSoup
import re
import time
import dateutil.parser as parser
from datetime import datetime
import datetime
import csv
import spacy
import re
from nltk.corpus import stopwords
import nltk
import emoji
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

# =====================================================================================================================================

# VARIABLES

nlp = spacy.load("en_core_web_sm")
contraction_colloq_dict = {"btw": "by the way", "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have"}

lemmatizer = nltk.stem.WordNetLemmatizer()

# Creating a storage.JSON file with authentication details
SCOPES = 'https://www.googleapis.com/auth/gmail.modify' # we are using modify and not readonly, as we will be marking the messages Read
store = file.Storage('storage.json') 
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
    creds = tools.run_flow(flow, store)
GMAIL = discovery.build('gmail', 'v1', http=creds.authorize(Http()))

user_id =  'me'
label_id_one = 'INBOX'
label_id_two = 'UNREAD'

# =====================================================================================================================================

# ALL FUNCTIONS

# Fetching message body
def extract_message_body(payload):
    if 'parts' in payload: # check if 'parts' key exists
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body']['data']
                clean_one = data.replace("-","+").replace("_","/") # decoding from Base64 to UTF-8
                clean_two = base64.b64decode(clean_one) # decoding from Base64 to UTF-8
                return clean_two.decode('utf-8') # return decoded text
            elif part['mimeType'] == 'multipart/alternative':
                return extract_message_body(part) # recursively search for the main message body within alternative parts
    elif 'body' in payload:
        data = payload['body']['data']
        clean_one = data.replace("-","+").replace("_","/") # decoding from Base64 to UTF-8
        clean_two = base64.b64decode(clean_one) # decoding from Base64 to UTF-8
        return clean_two.decode('utf-8') # return decoded text
    return '' # return empty string if message body not found

# Function to clean up message body
def clean_message_body(message_body):
    # Remove reply indicators and add a note indicating it's a reply
    cleaned_body = re.sub(r'(\s*>+\s*)+', '\n', message_body)
    # Remove line breaks and whitespace
    cleaned_body = cleaned_body.replace('\r', '').replace('\n', ' ').strip()  
    cleaned_body = cleaned_body.replace('<', '').replace('>', '').replace('*', '')
    cleaned_body = emoji.get_emoji_regexp().sub('', cleaned_body)
    cleaned_body = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_body)
    return cleaned_body

# Function to elongate contractions
def elongate_contractions(text, contraction_dict):
    # Regular expression pattern to find contractions
    pattern = re.compile(r'\b(' + '|'.join(contraction_dict.keys()) + r')\b')
    # Replace contractions with their full forms
    elongated_text = pattern.sub(lambda match: contraction_dict[match.group(0)] if match.group(0) in contraction_dict else match.group(0), text)
    return elongated_text

# Function to lemmatize text
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)

# Function to format date
def format_date(date_str):
    parsed_date = parser.parse(date_str)  # Parse the date string
    formatted_date = parsed_date.strftime("%A, %d %B %Y %I:%M %p")  # Format the date
    return formatted_date

# spell = SpellChecker()
# def spell_check(text):
#     corrected_text = []
#     # Split the text into words
#     words = text.split()
#     for word in words:
#         # Check if the word is misspelled
#         corrected_word = spell.correction(word)
#         # Append the corrected word if it's not None, otherwise keep the original word
#         corrected_text.append(corrected_word if corrected_word is not None else word)
#     # Join the corrected words back into a string
#     return ' '.join(corrected_text)


# =====================================================================================================================================

# MAIN CODE

# Getting last 10 emails
unread_msgs = GMAIL.users().messages().list(userId='me', labelIds=[label_id_one, label_id_two], maxResults=50).execute()

# We get a dictionary. Now reading values for the key 'messages'
mssg_list = unread_msgs.get('messages', [])

print ("Total unread messages retrieved: ", len(mssg_list))

final_list = []

for mssg in mssg_list:
    temp_dict = {}
    m_id = mssg['id'] # get id of individual message
    message = GMAIL.users().messages().get(userId=user_id, id=m_id).execute() # fetch the message using API
    payld = message['payload'] # get payload of the message 
    headr = payld['headers'] # get header of the payload

    # Extracting message details (Subject, Date, Sender, Snippet)
    for header in headr:
        name = header['name']
        value = header['value']
        if name == 'Subject':
            temp_dict['Subject'] = value
        elif name == 'Date':
            temp_dict['Date'] = value
        elif name == 'From':
            temp_dict['Sender'] = value
        elif name == 'Snippet':
            temp_dict['Snippet'] = value

    # Fetching message body
    temp_dict['Message_body'] = extract_message_body(payld)

    final_list.append(temp_dict)

print ("Total messages retrieved: ", len(final_list))

# Clean up message bodies and format dates
for item in final_list:
    item['Message_body'] = lemmatize_text(item['Message_body'])
    item['Message_body'] = clean_message_body(item['Message_body'])
    item['Date'] = format_date(item['Date'])
    item['Message_body'] = elongate_contractions(item['Message_body'], contraction_colloq_dict)
    item['Message_body'] = re.sub(r'\s+', ' ', item['Message_body'])


stopwords.words('english');
stop_words = set(stopwords.words('english'))

#Rmove stopwords
for item in final_list:
    doc = nlp(item['Message_body'])
    # Get tokens that are not stopwords
    filtered_tokens = [token.text for token in doc if token.text.lower() not in stop_words]
    # Join the filtered tokens back into a string
    item['Message_body'] = ' '.join(filtered_tokens)

# Print formatted messages
for item in final_list:
    print('='*100)
    print("Sender:", item['Sender'])
    print("Subject:", item['Subject'])
    print("Date:", item['Date'])
    print("Message Body:", item['Message_body'])

# Exporting the values as .csv
with open('last_10_emails.csv', 'w', encoding='utf-8', newline='') as csvfile: 
    fieldnames = ['Sender', 'Subject', 'Date', 'Message_body']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
    writer.writeheader()
    writer.writerows(final_list)

#  =====================================================================================================================================

# POS TAGGING
pos_tags = []
for item in final_list:
    cleaned_message = re.sub(r'[^\w\s]', '', item['Message_body'])
    # Process each cleaned message body
    doc = nlp(cleaned_message.lower())
    # Get part-of-speech tags for each token in the message body
    pos_tags_for_message = [(token.text, token.pos_) for token in doc]
    pos_tags.append(pos_tags_for_message)

print("\n\n===================================================================\n\n")
# Print POS tags for each message body
# for idx, tags in enumerate(pos_tags):
#     print('='*100)
#     print("Message Number:", idx+1)
#     for tag in tags:
#         print(tag[0], "-", tag[1])

# for i in pos_tags:
#     print(i)
#     print("\n\n")


# =====================================================================================================================================

# TF-IDF AND COUNT VECTORIZATION (K-MEANS)

# Extract message bodies for TF-IDF and Count Vectorization
messages = [item['Message_body'] for item in final_list]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000, min_df=0.2, stop_words='english', use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(messages)

# Count Vectorization
count_vectorizer = CountVectorizer(max_df=0.8, max_features=1000, min_df=0.2, stop_words='english')
count_matrix = count_vectorizer.fit_transform(messages)

# Determine the number of clusters based on the minimum of number of samples and number of clusters
num_clusters = min(len(final_list), 3)  # Adjust as needed

# K-means clustering using TF-IDF features
km_tfidf = KMeans(n_clusters=num_clusters)
km_tfidf.fit(tfidf_matrix)
clusters_tfidf = km_tfidf.labels_.tolist()

# K-means clustering using Count Vectorizer features
km_count = KMeans(n_clusters=num_clusters)
km_count.fit(count_matrix)
clusters_count = km_count.labels_.tolist()

# Adding cluster labels to final_list
for idx, item in enumerate(final_list):
    item['Cluster_TFIDF'] = clusters_tfidf[idx]
    item['Cluster_Count'] = clusters_count[idx]

# Print clusters
# print("\n\nTF-IDF Clusters:")
for i in range(num_clusters):
    # print(f"\nCluster {i}:")
    for idx, item in enumerate(final_list):
        if item['Cluster_TFIDF'] == i:
            pass
            # print(f"Sender: {item['Sender']} - Subject: {item['Subject']}")

print("\n\nCount Vectorizer Clusters:")
for i in range(num_clusters):
    # print(f"\nCluster {i}:")
    for idx, item in enumerate(final_list):
        if item['Cluster_Count'] == i:
            pass
            # print(f"Sender: {item['Sender']} - Subject: {item['Subject']}")


# Reduce dimensionality for visualization
pca = PCA(n_components=2)
tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())

# Further reduce dimensionality with t-SNE
perplexity = min(len(final_list) - 1, 50)  # Adjust the perplexity to be less than the number of samples
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
tfidf_tsne = tsne.fit_transform(tfidf_pca)

# Plot clusters
plt.figure(figsize=(10, 6))
for cluster_label in range(num_clusters):
    cluster_points = tfidf_tsne[np.array(clusters_tfidf) == cluster_label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}')

plt.title('t-SNE Visualization of TF-IDF Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()


# =====================================================================================================================================

# DOC2VEC (K-MEANS)

# Preprocess messages for Doc2Vec
tagged_data = [TaggedDocument(words=message.split(), tags=[str(idx)]) for idx, message in enumerate(messages)]

# Train Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Get document embeddings
doc_embeddings = [doc2vec_model.infer_vector(message.split()) for message in messages]

# Determine the number of clusters based on the minimum of number of samples and number of clusters
num_clusters = min(len(final_list), 3)  # Adjust as needed

# Perform K-means clustering
km_doc2vec = KMeans(n_clusters=num_clusters)
km_doc2vec.fit(doc_embeddings)
clusters_doc2vec = km_doc2vec.labels_.tolist()

# Adding cluster labels to final_list
for idx, item in enumerate(final_list):
    item['Cluster_Doc2Vec'] = clusters_doc2vec[idx]

# Print clusters
print("\n\nDoc2Vec Clusters:")
for i in range(num_clusters):
    print(f"\nCluster {i}:")
    for idx, item in enumerate(final_list):
        if item['Cluster_Doc2Vec'] == i:
            print(f"Sender: {item['Sender']} - Subject: {item['Subject']}")
