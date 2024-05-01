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

from typing import Union

import numpy as np


def square(value : Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return the square of a given number.

    This will accept any imput that support the ** operators, including
    integers, floating point numbers and numpy array.

    The function will raise a TypeError for incompatible types (e.g.,
    string).

    Arguments
    ---------
    value : number
        The input value to square.

    Return
    ------
        The square of the input value. : float
    """
    return value**2.
