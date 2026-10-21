import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def load_and_prepare(path='data/fake_and_real_news.csv', label_col='label'):
    df = pd.read_csv(path)
    if {'title','text'}.issubset(df.columns):
        df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
    else:
        df['content'] = df.iloc[:,0].astype(str)
    df['clean'] = df['content'].apply(clean_text)
    df['label_num'] = df[label_col].astype(int)
    return df[['content','clean','label_num']]
