import re

import en_core_web_md
import spacy
import pandas as pd

from arXiv_classification.utils import remove, is_string_series

def test_remove():
    """Unit test for the remove() function.
    """
    nlp = spacy.load("en_core_web_md")
    assert remove('hi_3 !? Hi, 374189 h5i math{hi} hi%& https:/website.com GitHub.com hi-\n', nlp) == 'hi'


def test_is_string_series():
    """Unit test for is_string_series() function.
    """
    d1 = {'a': 'hello', 'b': 'hi', 'c': 'python'}
    series1 = pd.Series(data=d1, index=['a', 'b', 'c'])
    d2 = {'a': 'hello', 'b': 'hi', 'c': 3}
    series2 = pd.Series(data=d2, index=['a', 'b', 'c'])    
    assert is_string_series(series1) == True
    assert is_string_series(series2) == False
    
