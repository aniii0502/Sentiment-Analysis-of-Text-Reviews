import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Cleans, tokenizes, and lemmatizes a single raw text string.
    This is the *exact* same pipeline used in the notebook.
    """
    
    text = text.lower()
    
    text = re.sub(r'<[^>]+>', ' ', text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    
    tokens = word_tokenize(text)
    
    clean_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in clean_tokens]
    
    return ' '.join(lemmatized_tokens)

if __name__ == '__main__':
    test_text = "This is a TEST review!! <br />It's running and feels awesome. 123"
    print(f"Original: {test_text}")
    print(f"Processed: {preprocess_text(test_text)}")