# import re

# import en_core_web_sm
# import spacy

# nlp = spacy.load("en_core_web_sm")

# def remove(text):
#     """
#     After tokenizing the text, remove punctuation and other characters.

#     Arguments
#     ---------
#        text: str, text to be processed
#        nlp: Spacy model

#     Returns
#     -------
#        filtered_text: str, processed text
#     """
#     # Apply the natural language Spacy model to the input text.
#     tokens = nlp(text)
#     # Transform to lowercase and then split the text.
#     tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens] 
#     # Remove the tokens that contains those strings
#     B = ["github.com", "http"] 
#     filtered_text = [word for word in tokens if not any(bad in word for bad in B)] # filtered_text must be a list of words
#     # Remove the numeric characters.
#     filtered_text = " ".join(c for c in filtered_text if c.isalpha() or c.isspace()) 
#     filtered_text = re.sub('(?:\s)http[^, ]*', '', filtered_text) # remove the words that begin with http (filtered_text must be a string)
#     filtered_text = filtered_text.replace("- ", "")  # join the words that are hyphenated
#     return filtered_text

import numpy as np

from arXiv_classification.utils_prova import square

def test_square():
    """Unit test for the square() function.
    """
    assert square(2.) == 4.
    assert square(-2.) == 4.
    assert square(0.) == 0.
    assert square(2) == 4
    assert square(-2) == 4
    assert square(0) == 0
    assert np.allclose(square(np.ones(100)), np.ones(100))
