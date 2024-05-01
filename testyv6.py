import base64
import re
import emoji
from email.utils import parsedate_to_datetime
import spacy
from nltk.corpus import stopwords
import nltk
from apiclient.discovery import build
from httplib2 import Http
from oauth2client.file import Storage
from oauth2client.client import flow_from_clientsecrets
from oauth2client.tools import run_flow
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Load NLTK and SpaCy resources
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Google API setup
SCOPES = 'https://www.googleapis.com/auth/gmail.modify'
store = Storage('storage.json')
creds = store.get()
if not creds or creds.invalid:
    flow = flow_from_clientsecrets('credentials.json', SCOPES)
    creds = run_flow(flow, store)
GMAIL = build('gmail', 'v1', http=creds.authorize(Http()))

# BERT model setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
label_map = {0: 'General', 1: 'Promotion', 2: 'Work', 3: 'Event'}

def extract_message_body(payload):
    body = ''
    if 'parts' in payload:
        for part in payload['parts']:
            body += extract_message_body(part)
    else:
        body_data = payload.get('body', {}).get('data', '')
        if body_data:
            body = base64.urlsafe_b64decode(body_data).decode('utf-8')
    return body

def clean_message_body(message):
    message = emoji.get_emoji_regexp().sub(r'', message)
    message = re.sub(r'https?://\S+', '', message)
    message = re.sub(r'[\r|\n|\r\n]+', ' ', message).strip()
    # Remove stopwords and lemmatize
    tokens = nlp(message)
    return ' '.join([token.lemma_ for token in tokens if token.text.lower() not in stop_words and not token.is_punct])

def plot_clusters(data_matrix, cluster_labels, title):
    pca = PCA(n_components=2)  # Reduce dimensions to 2 for visualization
    reduced_data = pca.fit_transform(data_matrix.toarray())  # Convert sparse matrix to array before transforming
    plt.figure(figsize=(8, 6))  # Set the figure size
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    plt.show()


def fetch_and_classify_emails():
    response = GMAIL.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=50).execute()
    emails = []
    texts = []
    for msg in response.get('messages', []):
        txt = GMAIL.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        payload = txt.get('payload', {})
        subject = next(header['value'] for header in payload.get('headers', []) if header['name'].lower() == 'subject')
        from_ = next(header['value'] for header in payload.get('headers', []) if header['name'].lower() == 'from')

        body = extract_message_body(payload)
        body = clean_message_body(body)

        inputs = tokenizer(body, return_tensors="tf", truncation=True, padding=True, max_length=512)
        outputs = model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_label = label_map[tf.argmax(predictions, axis=1).numpy()[0]]

        emails.append({
            'subject': subject,
            'from': from_,
            'body': body,
            'category': predicted_label
        })
        texts.append(body)

    # Cluster emails using TF-IDF and Count Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    km_tfidf = KMeans(n_clusters=4)  # Adjust number of clusters as necessary
    km_tfidf.fit(tfidf_matrix)

    count_vectorizer = CountVectorizer(max_features=1000)
    count_matrix = count_vectorizer.fit_transform(texts)
    km_count = KMeans(n_clusters=4)  # Adjust number of clusters as necessary
    km_count.fit(count_matrix)

    # Add cluster information to emails
    for idx, email in enumerate(emails):
        email['cluster_tfidf'] = km_tfidf.labels_[idx]
        email['cluster_count'] = km_count.labels_[idx]

    # Visualization of the clusters
    plot_clusters(tfidf_matrix, km_tfidf.labels_, 'TF-IDF Clustering Visualization')
    plot_clusters(count_matrix, km_count.labels_, 'Count Vectorization Clustering Visualization')

    return emails



def parse_query(text):
    # Process the text with NLP tool to parse it
    doc = nlp(text)
    # List to hold significant words for search
    keywords = []
    # Ignored parts of speech: determiners, conjunctions, pronouns, auxiliary verbs, etc.
    ignored_pos = {'DET', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'AUX', 'ADP', 'SCONJ'}
    # Extract entities and nouns, ignoring less significant words
    for token in doc:
        if token.pos_ not in ignored_pos and token.dep_ not in ['aux', 'auxpass']:
            keywords.append(token.text.lower())
    # Join the filtered keywords to form a search phrase
    search_phrase = ' '.join(keywords)
    return {'intent': 'search', 'query': text, 'search_phrase': search_phrase, 'entities': {ent.label_: ent.text for ent in doc.ents}, 'keywords': keywords}


def search_emails(query, emails):
    results = []
    query_lower = query.lower()  # Convert the query to lowercase for case-insensitive comparison
    for email in emails:
        # Check if the exact phrase is present in the email body
        if query_lower in email['body'].lower():
            # Extract dates from the email body
            doc = nlp(email['body'])
            dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE' or ent.label_ == 'TIME']
            email['dates'] = dates
            results.append(email)
    return results

def format_response(results):
    if not results:
        return "No relevant emails found."
    response = "Here are the relevant details:\n"
    for email in results:
        date_info = ', '.join(email['dates']) if email['dates'] else "No specific date mentioned"
        response += f"From: {email['from']}, Subject: {email['subject']}, Date/Time: {date_info}\n\n"
    return response


def chatbot():
    print("Hello! Ask me about any events or topics in your emails, such as 'What about philosophy movie night?'")
    emails = fetch_and_classify_emails()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        query_details = parse_query(user_input)
        results = search_emails(query_details['search_phrase'], emails)  # Use the refined search phrase
        if not results:
            print("No relevant emails found concerning: " + user_input)
        else:
            print("Here are the emails related to your query:")
            for email in results:
                date_info = ', '.join(email['dates']) if email['dates'] else "No specific date mentioned"
                print(f"From: {email['from']}, Subject: {email['subject']}, Date/Time: {date_info}\n")



if __name__ == "__main__":
    chatbot()
