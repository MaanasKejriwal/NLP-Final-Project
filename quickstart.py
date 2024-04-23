'''
Reading GMAIL using Python

'''

'''
This script does the following:
- Go to Gmal inbox
- Find and read all the unread messages
- Extract details (Date, Sender, Subject, Snippet, Body) and export them to a .csv file / DB
- Mark the messages as Read - so that they are not read again 
'''

'''
Before running this script, the user should get the authentication by following 
the link: https://developers.google.com/gmail/api/quickstart/python
Also, client_secret.json should be saved in the same directory as this file
'''

# Importing required libraries
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

#Getting lasst 10 emails

#Getting last 10 emails
unread_msgs = GMAIL.users().messages().list(userId='me', labelIds=[label_id_one, label_id_two], maxResults=10).execute()

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
    try:
        if 'parts' in payld: # check if 'parts' key exists
            mssg_parts = payld['parts'] # fetching the message parts
            part_one = mssg_parts[0] # fetching first element of the part 
            if 'body' in part_one: # check if 'body' key exists
                part_body = part_one['body'] # fetching body of the message
                if 'data' in part_body: # check if 'data' key exists
                    part_data = part_body['data'] # fetching data from the body
                    clean_one = part_data.replace("-","+").replace("_","/") # decoding from Base64 to UTF-8
                    clean_two = base64.b64decode(clean_one) # decoding from Base64 to UTF-8
                    soup = BeautifulSoup(clean_two, "html.parser")
                    mssg_body = soup.get_text() # extracting text from HTML
                    temp_dict['Message_body'] = mssg_body
                else:
                    temp_dict['Message_body'] = '' # set message body to empty if 'data' key is missing
        else:
            temp_dict['Message_body'] = '' # set message body to empty if 'parts' key is missing
    except Exception as e:
        print(f"Error fetching message body: {e}")

    final_list.append(temp_dict)

print ("Total messages retrieved: ", len(final_list))
print(final_list)

message_bodies = []

for item in final_list:
    message_bodies.append([item['Message_body']])

print(message_bodies)
 
# Exporting the values as .csv
with open('last_10_emails.csv', 'w', encoding='utf-8', newline='') as csvfile: 
    fieldnames = ['Sender', 'Subject', 'Date', 'Snippet', 'Message_body']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
    writer.writeheader()
    writer.writerows(final_list)
