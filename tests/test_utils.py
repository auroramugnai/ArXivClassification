import re

# import en_core_web_sm
# import spacy

from arXiv_classification.utils_prova import remove

def test_remove():
    """Unit test for the remove() function.
    """
    # nlp = spacy.load("en_core_web_sm")
    assert remove('ciao_3') == 'ciao'
    assert remove('mi chiamo Chiara! Tu come ti chiami?') == 'mi chiamo chiara tu come ti chiami'
    assert remove('prova%$ https:/sito.com') == 'prova'

