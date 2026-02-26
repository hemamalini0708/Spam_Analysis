import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import sklearn
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
nltk.download('punkt',quiet=True)
nltk.download('wordnet')
nltk.download('stopwords')
import tensorflow
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from log_file import Logger
logger = Logger.get_logs('main')
import warnings
warnings.filterwarnings('ignore')

class sent_analysis:
    def __init__(self):
        self.df=pd.read_csv(r'D:\\Users\\geeth\\PycharmProjects\\Spam_Analysis\\spam_.csv',encoding='latin_1')
        #logger.info(self.df.head(5))
        # logger.info(self.df['sentiment'].value_counts())
    def model_loading(self):
        try:
            with open('spam_review.pkl', 'rb') as f:
                self.m = pickle.load(f)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Performance Error : {er_lin.tb_lineno} : due to {er_msg}")
    def model_testing(self):
        labels = ['ham', 'spam']
        dic_size = 5500
        review = ['Hello Batta,We’re writing to let you know that you were not selected for the role of AI Research & Development Graduate Intern at Johnson & Johnson.Your credentials are aligned with our needs but were not an exact fit for this job. We appreciate your effort behind your application and your interest in contributing. News like this is always disappointing, no matter where you are in the job search process, but we hope you are not discouraged. Indeed, you may be a perfect fit for a different role.We invite you to stay in touch, as new opportunities are constantly developing that may be a perfect fit for you. Please check our Careers site  for the newest openings. You can also join our Global Talent Hub, where we keep in touch with people around the world who share our passion for bold innovations and are inspired by our mission of changing the trajectory of human health.We wish you the best of luck as you continue your search. We hope that this won’t be the last time we cross paths, because there’s no end to the lasting impact we could make together.Best Regards,Johnson & Johnson Talent Acquisition Team']
        text = review[0].lower()
        text = ''.join([i for i in text if i not in string.punctuation])
        text = ' '.join([lemma.lemmatize(i) for i in text.split() if i not in stopwords.words('english')])
        v = [one_hot(i, dic_size) for i in [text]]
        p = pad_sequences(v, maxlen=953, padding='post')
        print(labels[np.argmax(self.m.predict(p))])

if __name__ == "__main__":
    obj=sent_analysis()
    obj.model_loading()
    obj.model_testing()