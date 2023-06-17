import string
import pandas as pd
import re
import contractions
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class SentimentAnalysis(object):

    def __init__(self):
        cols = ['id', 'sentiment', 'sentiment_confidence', 'negativereason', 'negativereason_confidence', 'airline',
                'airline_sentiment_gold', 'name', 'negativereason_gold', 'retweet_count', 'text', 'tweet_coord', 'tweet_created',
                'tweet_location', 'user_timezone']
        self.data = pd.read_csv('Tweets.csv', encoding='ISO-8859-1', header=0, names=cols)
        self.split_dataset()

    def preprocess(self):
        #self.data['sentiment'] = LabelEncoder().fit_transform(self.data['sentiment'])
        urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        userPattern = '@[^\s]+'
        stemmer = PorterStemmer()
        for i in range(len(self.data['text'])):
            # Lower case all text
            self.data['text'][i] = self.data['text'][i].lower()
            # Expand Contractions
            self.data['text'][i] = contractions.fix(self.data['text'][i])
            # Remove all urls
            self.data['text'][i] = re.sub(urlPattern, '', self.data['text'][i])
            # Remove all @username
            self.data['text'][i] = re.sub(userPattern, '', self.data['text'][i])
            # Remove punctuations
            self.data['text'][i] = self.data['text'][i].translate(str.maketrans('', '', string.punctuation))
            # Stem Words
            self.data['text'][i] = stemmer.stem(self.data['text'][i])

    def transform(self):
        self.pipe = Pipeline([
                              ('vect', CountVectorizer(stop_words='english'))
                            ])

        self.X_Tr = self.pipe.fit_transform(self.X_Tr).toarray()
        self.X_Ts = self.pipe.transform(self.X_Ts).toarray()

    def split_dataset(self):
        # Preprocess text
        self.preprocess()
        # Split into training and testing data
        self.X = self.data['text']
        self.Y = self.data['sentiment']
        self.X_Tr, self.X_Ts, self.Y_Tr, self.Y_Ts = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        self.transform()

    def NB_train(self):
        self.NB_model = ComplementNB()
        self.NB_model.fit(self.X_Tr, self.Y_Tr)
        self.NB_accuracy_Tr = self.NB_model.score(self.X_Tr, self.Y_Tr)
        print("NB Train Accuracy: {:.2f}%".format(self.NB_accuracy_Tr))
        print(f"NB Train Classifier: \n\n {classification_report(self.Y_Tr, self.NB_model.predict(self.X_Tr))}")
        plot_confusion_matrix(self.NB_model, self.X_Tr, self.Y_Tr)
        plt.show()

    def NB_test(self):
        self.NB_accuracy_Ts = self.NB_model.score(self.X_Ts, self.Y_Ts)
        print("NB Test Accuracy = {:.2f}%".format(self.NB_accuracy_Ts))
        print(f"NB Test Classifier: \n\n {classification_report(self.Y_Ts, self.NB_model.predict(self.X_Ts))}")
        plot_confusion_matrix(self.NB_model, self.X_Ts, self.Y_Ts)
        plt.show()

    def LR_train(self):
        self.LR_model = LogisticRegression()
        self.LR_model.fit(self.X_Tr, self.Y_Tr)
        self.LR_accuracy_Tr = self.LR_model.score(self.X_Tr, self.Y_Tr)
        print("LR Train Accuracy: {:.2f}%".format(self.LR_accuracy_Tr))
        print(f"LR Train Classifier: \n\n {classification_report(self.Y_Tr, self.LR_model.predict(self.X_Tr))}")
        plot_confusion_matrix(self.LR_model, self.X_Tr, self.Y_Tr)
        plt.show()

    def LR_test(self):
        self.LR_accuracy_Ts = self.LR_model.score(self.X_Ts, self.Y_Ts)
        print("LR Test Accuracy = {:.2f}%".format(self.LR_accuracy_Ts))
        print(f"LR Test Classifier: \n\n {classification_report(self.Y_Ts, self.LR_model.predict(self.X_Ts))}")
        plot_confusion_matrix(self.LR_model, self.X_Ts, self.Y_Ts)
        plt.show()

    def SVM_train(self):
        self.SVM_model = svm.SVC(kernel='linear')
        self.SVM_model.fit(self.X_Tr, self.Y_Tr)
        self.SVM_accuracy_Tr = self.SVM_model.score(self.X_Tr, self.Y_Tr)
        print("SVM Train Accuracy: {:.2f}%".format(self.SVM_accuracy_Tr))
        print(f"SVM Train Classifier: \n\n {classification_report(self.Y_Tr, self.SVM_model.predict(self.X_Tr))}")
        plot_confusion_matrix(self.SVM_model, self.X_Tr, self.Y_Tr)
        plt.show()

    def SVM_test(self):
        self.SVM_accuracy_Ts = self.SVM_model.score(self.X_Ts, self.Y_Ts)
        print("SVM Test Accuracy = {:.2f}%".format(self.SVM_accuracy_Ts))
        print(f"SVM Test Classifier: \n\n {classification_report(self.Y_Ts, self.SVM_model.predict(self.X_Ts))}")
        plot_confusion_matrix(self.SVM_model, self.X_Ts, self.Y_Ts)
        plt.show()

    def RF_train(self):
        self.RF_model = RandomForestRegressor(n_estimators=1000, random_state=42)
        self.RF_model.fit(self.X_Tr, self.Y_Tr)
        self.RF_accuracy__Tr = self.RF_model.score(self.X_Tr, self.Y_Tr)
        print("RF Train Accuracy: {:.2f}%".format(self.RF_accuracy_Tr))
        print(f"RF Train Classifier: \n\n {classification_report(self.Y_Tr, self.RF_model.predict(self.X_Tr))}")
        plot_confusion_matrix(self.SVM_model, self.X_Tr, self.Y_Tr)
        plt.show()

    def RF_test(self):
        self.RF_accuracy_RF = self.RF_model.score(self.X_Ts, self.Ts)
        print("RF Test Accuracy = {:.2f}%".format(self.RF_accuracy_Ts))
        print(f"RF Test Classifier: \n\n {classification_report(self.Y_Ts, self.RF_model.predict(self.X_Ts))}")
        plot_confusion_matrix(self.RF_model, self.X_Ts, self.Y_Ts)
        plt.show()

    def complete_list(self):
        print("NB Train Accuracy: {:.2f}%".format(self.NB_accuracy_Tr))
        print("NB Test Accuracy = {:.2f}%".format(self.NB_accuracy_Ts))
        #print("LR Train Accuracy: {:.2f}%".format(self.LR_accuracy_Tr))
        #print("LR Test Accuracy = {:.2f}%".format(self.LR_accuracy_Ts))
        #print("SVM Train Accuracy: {:.2f}%".format(self.SVM_accuracy_Tr))
        #print("SVM Test Accuracy = {:.2f}%".format(self.SVM_accuracy_Ts))
        #print("RF Train Accuracy: {:.2f}%".format(self.RF_accuracy_Tr))
        #print("RF Test Accuracy = {:.2f}%".format(self.RF_accuracy_Ts))



