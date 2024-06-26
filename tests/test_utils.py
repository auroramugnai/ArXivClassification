import en_core_web_md
import spacy
import pandas as pd

from ArXivClassification.utils import text_cleaner, is_string_series

def test_text_cleaner():
    """Unit test for the text_cleaner() function.
    """
    nlp = spacy.load("en_core_web_md")
    assert text_cleaner('This is a test: h3llo $hello$ hello% test_hello', nlp) == 'test'
    assert text_cleaner('www.website.com https://website hel- lo', nlp) == 'hello'


def test_is_string_series():
    """Unit test for is_string_series() function.
    """
    d1 = {'a': 'This', 'b': 'is', 'c': 'Python'}
    d2 = {'a': 'My age', 'b': 'is', 'c': 23}
    d3 = {'a': ['Hello', 'my'], 'b': ['name', 'is'], 'c': ['Pippo', '!']}
    d4 = {'a': "['Hello', 'my']", 'b': "['name', 'is']", 'c': "['Pippo', '!']"}
    assert is_string_series(pd.Series(data=d1, index=['a', 'b', 'c'])) == True
    assert is_string_series(pd.Series(data=d2, index=['a', 'b', 'c'])) == False
    assert is_string_series(pd.Series(data=d3, index=['a', 'b', 'c'])) == False
    assert is_string_series(pd.Series(data=d4, index=['a', 'b', 'c'])) == True
    
