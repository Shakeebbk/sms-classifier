'''
# SMS

Hello world
'''
import streamlit as st

import numpy as np
import pandas as pd

from sklearn.model_selection  import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import warnings
warnings.filterwarnings("ignore")

st.title("SMS classifier")

def get_model():
#    # reading the training data
#    docs = pd.read_table('SMSSpamCollection', header=None, names=['Class', 'sms'])
#    docs['Class'] = docs.Class.map({'ham':0, 'spam':1})
#
#    X = docs.sms
#    y = docs.Class
#    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.01)
#
#    vect = CountVectorizer(stop_words='english')
#
#    vect.fit(X_train)
#
#    X_train_transformed = vect.transform(X_train)
#    X_test_transformed = vect.transform(X_test)
#
#    # training the NB model and making predictions
#    mnb = MultinomialNB()
#
#    # fit
#    mnb.fit(X_train_transformed,y_train)

    from sklearn.externals import joblib
    model = open('mnb.pkl','rb')
    mnb = joblib.load(model)

    vect_file = open('vect.pkl', 'rb')
    vect = joblib.load(vect_file)

    return vect, mnb

'''
Let's classify your messages in to spam or ham.
'''
sms = st.text_input("Enter your SMS here:")
if sms:
    # st.write("You entered:", sms)
    vect, model = get_model()

    sample_transformed = vect.transform([sms])

    predicted_class = model.predict(sample_transformed)
    # st.write(predicted_class)
    if predicted_class:
        st.write("It's a SPAM SMS")
    else:
        st.write("It's a ham SMS")

    '''
    *If the prediction is wrong, please help improve by clicking the button.*
    '''
    if st.button('Submit'):
        if predicted_class:
            f_c = "ham/new_data.txt"
        else:
            f_c = "spam/new_data.txt"
        existing_lines = []
        with open(f_c, 'r') as f:
            existing_lines = f.readlines()

        with open(f_c, 'a') as f:
            if sms+'\n' not in existing_lines:
                f.write(sms+'\n')
        '''
        Thanks for helping out.
        Try a new word now.
        '''

