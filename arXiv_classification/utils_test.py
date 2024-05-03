from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd

# Modules to import only for type checking.
if TYPE_CHECKING:
    import spacy

def is_string_series(s: pd.Series) -> bool:
    """Checks whether s series contains only strings.
    """
    if isinstance(s.dtype, pd.StringDtype):
        # String series.
        return True


def text_cleaner(text: str, nlp: spacy.lang.en.English) -> str:
    """
    After joining interrupted words and tokenizing the text, 
    lemmatize, remove bad words, special characters, punctuation
    and Spacy stopwords.

    Arguments
    ---------
       text: str, text to be cleaned
       nlp: Spacy model

    Returns
    -------
       clean_tokens: str, cleaned text
    """
    # Join interrupted words.
    text = text.replace("- ", "")  

    # Tokenize.
    tokens = nlp(text) # list of words

    # Lemmatize and transform to lowercase.
    #"-PRON-" is used as the lemma for all pronouns such as I, me, their ...
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" 
              else word.lower_ for word in tokens] 

    # Remove the tokens that contain any string in badlist.
    badlist = [".com", "http"] 
    clean_tokens = [word for word in tokens 
                    if not any(bad in word for bad in badlist)]
  
    # Remove stopwords.
    from spacy.lang.en.stop_words import STOP_WORDS
    clean_tokens = [word for word in clean_tokens 
                    if not word in STOP_WORDS] 

    # Keep only characters that are alphabet letters or spaces.
    clean_tokens = " ".join(c for c in clean_tokens 
                            if c.isalpha() or c.isspace()) 
  
    return clean_tokens
