# import re

# from arXiv_classification.utils_prova import remove

# def test_remove():
#     """Unit test for the remove() function.
#     """
#     # nlp = spacy.load("en_core_web_sm")
#     assert remove('ciao_3 ciao') == 'ciao'
#     assert remove('mi chiamo Chiara! Tu come ti chiami?') == 'mi chiamo chiara tu come ti chiami'
#     assert remove('prova%$ https:/sito.com prova') == 'prova'

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
