import pandas as pd
import numpy as np
import textblob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps = PorterStemmer()
true_hindi=pd.read_csv(r"C:\Users\Manas Pati Tripathi\Downloads\basic\true_news_basic.csv")
fake_hindi=pd.read_csv(r"C:\Users\Manas Pati Tripathi\Downloads\basic\fake_news_basic.csv")
import csv
fake_english=pd.read_csv(r"C:\Users\Manas Pati Tripathi\Downloads\News-_dataset\Fake.csv")
true_english=pd.read_csv(r"C:\Users\Manas Pati Tripathi\Downloads\News-_dataset\True.csv")
true_hindi['tf']=1
fake_hindi['tf']=0
true_english['tf']=1
fake_english['tf']=0
english=pd.concat([fake_english,true_english])
hindi=pd.concat([fake_hindi,true_hindi])
english.reset_index(inplace=True)
hindi.reset_index(inplace=True)
english.drop('subject', axis=1, inplace=True)
english.drop('date', axis=1, inplace=True)
hindi.drop(['index','Unnamed: 0'],axis=1,inplace=True)
text_list=[]
text_list_h=[]
Yh=[]
Ye=[]
import random
from translate import Translator

translator = Translator(to_lang="en", from_lang="hi")
hien=[]
rand_list=[]
n=2010
for i in range(n):
    rand_list.append(random.randint(0,44898))
for j in rand_list:
    Ye.append(english['tf'][j])
    text_list.append(english['text'][j])
print(len(Ye))     
r=[]

for ii in range(n):
    r.append(random.randint(0,2010))

for k in r:
    Yh.append(hindi['tf'][k])  
    text_list_h.append(hindi['short_description'][k])
    hien.append(translator.translate(hindi['short_description'][k]))

print(len(Yh))



#SENTIMENT ANALYSIS PART 

from textblob import TextBlob




def perform_sentiment_analysis(text):
    analysis = TextBlob(text)

    # Determine sentiment polarity and subjectivity
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    # Classify sentiment
    sentiment_class = "Neutral"
    if polarity > 0:
        sentiment_class = "Positive"
    elif polarity < 0:
        sentiment_class = "Negative"

    return sentiment_class, polarity, subjectivity


sentiment=[]
polar_score=[]
subjective_score=[]


# Perform sentiment analysis for each article
sentiment_results = [perform_sentiment_analysis(article) for article in text_list]


for idx, (sentiment_class, polarity, subjectivity) in enumerate(sentiment_results, start=0):
    sentiment.append(sentiment_class)
    polar_score.append(polarity)
    subjective_score.append(subjectivity)
english['sentiment']=sentiment
english['polarity']=polar_score
english['subjectivity']=subjective_score



import nltk
#from vaderSentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


#hindi.drop(['sentiment','polar'],axis=1,inplace=True)


sentiment_h=[]
polar_score_h=[]
score=[]
hien=[]
def sent(n):
  if n > 0.05:
    sentiment_label = "Positive"
  elif n < -0.05:
    sentiment_label = "Negative"
  else:
    sentiment_label = "Neutral"
  return sentiment_label
# Example news articles



for i,sentence in enumerate(hien):

    score.append(analyzer.polarity_scores(sentence))
    polar_score_h.append(score[i]['compound'])
    sentiment_h.append(sent(polar_score_h[i]))
hindi['sentiment']=sentiment_h
hindi['polar']=polar_score_h


#TEXT PREPROCESSING 

   

for i in range(0, len(text_list)):
    review = re.sub('[^a-zA-Z]', ' ', text_list[i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    english_corpus.append(review)
print(len(english_corpus))




stopwords_hi = ['तुम','मेरी','मुझे','क्योंकि','हम','प्रति','अबकी','आगे','माननीय','शहर','बताएं','कौनसी','क्लिक','किसकी','बड़े','मैं','and','रही','आज','लें','आपके','मिलकर','सब','मेरे','जी','श्री','वैसा','आपका','अंदर', 'अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर','करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वर्ग', 'वह', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'वग़ैरह', 'संग', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने', 'अपनि', 'जेसे', 'होति', 'सभि', 'तिंहों', 'इंहों', 'दवारा', 'इसि', 'किंहें', 'थि', 'उंहों', 'ओर', 'जिंहें', 'वहिं', 'अभि', 'बनि', 'हि', 'उंहिं', 'उंहें', 'हें', 'वगेरह', 'एसे', 'रवासा', 'कोन', 'निचे', 'काफि', 'उसि', 'पुरा', 'भितर', 'हे', 'बहि', 'वहां', 'कोइ', 'यहां', 'जिंहों', 'तिंहें', 'किसि', 'कइ', 'यहि', 'इंहिं', 'जिधर', 'इंहें', 'अदि', 'इतयादि', 'हुइ', 'कोनसा', 'इसकि', 'दुसरे', 'जहां', 'अप', 'किंहों', 'उनकि', 'भि', 'वरग', 'हुअ', 'जेसा', 'नहिं']
#punctuations = ['nn','n', '।','/', '`', '+', '\', '"', '?', '▁(', '$', '@', '[', '_', "'", '!', ',', ':', '^', '|', ']', '=', '%', '&', '.', '(', ')', "#", '*', '', ';', '-', '{','}','|']
to_be_removed = stopwords_hi  
hindi_corpus = []
for i in range(0, len(text_list_h)):
    review=text_list_h[i]
    review = review.split()
    review=[ps.stem(ele) for ele in review if ele not in (to_be_removed)]
    review=' '.join(review)
    hindi_corpus.append(review)
print(len(hindi_corpus))
corpus=hindi_corpus+english_corpus
Y=Yh+Ye


#VECTORISATION

from sklearn.model_selection import train_test_split
X_train, X_test, Y_t, Y_ts = train_test_split(corpus,Y, test_size=0.33, random_state=0)


Y_train=np.array(Y_t)
Y_test=np.array(Y_ts)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
Xt=tfidf_v.fit_transform(X_train).toarray()

Xs=tfidf_v.transform(X_test).toarray()

print(len(Xt))

print(len(Y_train))

#ENSEMBLE MODEL TRAINING

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



# Preprocess the data (tokenization, TF-IDF vectorization, etc.)
# You can customize this preprocessing based on your dataset.

# Split the data into training and testing sets


# Build and train the Multinomial Naive Bayes model

mnb_model = MultinomialNB()
mnb_model.fit(Xt, Y_train)
mnb_predictions = mnb_model.predict(Xs)
print(mnb_predictions)
# Build and train the Bidirectional LSTM model
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_sequences, maxlen=100, padding='post')
X_test_pad = pad_sequences(X_test_sequences, maxlen=100, padding='post')
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
lstm_model.add(Bidirectional(LSTM(64)))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
lstm_model.fit(X_train_pad, Y_train, epochs=5, batch_size=64)
lstm_pred = (lstm_model.predict(X_test_pad) > 0.5).astype("int32")
lstm_predictions = [int(item) for sublist in lstm_pred for item in sublist]

# Ensemble the predictions from both models
ensemble_predictions = np.round((mnb_predictions + lstm_predictions) / 2).astype(int)
print(ensemble_predictions)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
conf_matrix = confusion_matrix(Y_test, ensemble_predictions)

# Print the classification report
classification_rep = classification_report(Y_test, ensemble_predictions)
print(classification_rep)
# Visualize the confusion matrix
def plot_confusion_matrix(conf_matrix, labels):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Plot the confusion matrix
labels = ['Real', 'Fake']
plot_confusion_matrix(conf_matrix, labels)


# Evaluate the ensemble model
ensemble_accuracy = accuracy_score(Y_test, ensemble_predictions)
print(ensemble_accuracy)
