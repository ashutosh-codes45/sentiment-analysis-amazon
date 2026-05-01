import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

stops = set(stopwords.words('english'))

url="https://drive.google.com/uc?export=download&id=1d5AYSaHvqdTMLEdzBwqaNW90fUm-tZhU"
df=pd.read_csv(url, engine='python', on_bad_lines='skip')
# print(df.shape)
df = df[['Rating', 'Review Text']]

# df[['Review Text']].isna() == True
# df.isnull().sum()

df['Rating'] = df['Rating'].str.extract(r'Rated (\d) out of 5 stars').astype(float)

# separate as 1 -> +ve, 0 -> -ve

df = df[df['Rating']!= 3]
df['sentiment'] = df['Rating'].apply(lambda x : 1 if x>=4 else 0)
# print(df[df['sentiment']==1.0])

# Text Preprocessing :-

# lowercasing all the Strings under 'Review Text' :-
df['Review Text'] = df['Review Text'].str.lower()
# print(df)

# removing punctuations :-
df['Review Text'] = df['Review Text'].str.translate(str.maketrans('', '', string.punctuation))

# removing stopwords :-
df['Review Text'] = df['Review Text'].fillna('').astype(str)
df['Review Text'] = df['Review Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stops]))

# lemmatization of the 'Review Text' column :-

lemmatizer = WordNetLemmatizer()

def get_pos(tag) : # mapping pos_tags for precise lemmatization
  if tag.startswith('J') :
    return wordnet.ADJ
  elif tag.startswith('V') :
    return wordnet.VERB
  elif tag.startswith('N') :
    return wordnet.NOUN
  elif tag.startswith('R') :
    return wordnet.ADV
  else :
    return wordnet.NOUN

def lemmatize_txt(text) :
  words = nltk.word_tokenize(text)
  tagged = nltk.pos_tag(words)

  return " ".join([lemmatizer.lemmatize(word, get_pos(tag)) for word, tag in tagged])

df['Review Text'] = df['Review Text'].apply(lemmatize_txt)
# print(df)

# Feature Extraction (Vectorization of Text) :-

def tf_idf() :
  
  tfidf = TfidfVectorizer(max_features=2000)
  X = tfidf.fit_transform(df['Review Text'])
  y = df['sentiment'] # output
  # print(X.shape)

  return X, y, tfidf