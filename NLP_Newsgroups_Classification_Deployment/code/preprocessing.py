import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def preprocess_text_spacy(text):
    # Remove metadata headers
    text = re.sub(r'^.*?:.*?\n', '', text, flags=re.MULTILINE)

    # Remove emails
    text = re.sub(r'\S*@\S*\s?', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove 'GMT'
    text = re.sub(r'GMT', '', text)

    # Remove extra whitespace and newline characters
    text = re.sub(r'\s+', ' ', text)

    # Apply spaCy tokenization and lemmatization
    doc = nlp(text)

    # Lemmatize tokens, lowercase and remove stopwords, punctuation and special characters
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.isalnum()]

    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text
