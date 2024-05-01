import streamlit as st
from functions import fetch_and_classify_emails, parse_query, search_emails, summarize_email_from_sender, summarize_last_x_emails

# Load emails initially or reload on demand

def load_emails():
    return fetch_and_classify_emails()

def display_email_results(results):
    for email in results:
        st.text(f"From: {email['from']}\nSubject: {email['subject']}\nDate/Time: {'; '.join(email.get('dates', ['No specific date mentioned']))}\n")
        st.write("---")

emails = load_emails()

st.title("Email Query and Summarization Tool")

query_input = st.text_input("Enter your query:", "Type here...")

if st.button("Search"):
    query_details = parse_query(query_input)
    results = search_emails(query_details, emails)
    if not results:
        st.write("No relevant emails found concerning your query.")
    else:
        if query_details['intent'] == 'search_deadlines':
            st.subheader("Upcoming Deadlines:")
            display_email_results(results)
        else:
            st.subheader("Emails related to your query:")
            display_email_results(results)

elif st.button("Summarize Last X Emails"):
    num_emails = st.slider("Select the number of emails to summarize:", min_value=1, max_value=5, value=3)
    summary = summarize_last_x_emails(emails, num_emails)
    st.write("Combined Summary of Last Emails:")
    st.write(summary)

elif st.button("Summarize Emails from Specific Sender"):
    sender_name = st.text_input("Enter the sender's name:")
    if st.button("Summarize"):
        result = summarize_email_from_sender(emails, sender_name)
        if isinstance(result, str):
            st.write(result)
        else:
            st.write(f"From: {result['from']}, Subject: {result['subject']}, Summary: {result['summary']}")

# Optional: add a button to refresh email data
if st.button("Reload Emails"):
    emails = load_emails()
    st.write("Emails reloaded successfully!")
