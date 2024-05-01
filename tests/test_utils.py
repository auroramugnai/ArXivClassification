import re

import en_core_web_md
import spacy

from arXiv_classification.utils import remove

def test_remove():
    """Unit test for the remove() function.
    """
    nlp = spacy.load("en_core_web_md")
    assert remove('ciao_3', nlp) == 'ciao'
    assert remove('mi chiamo Chiara! Tu come ti chiami?', nlp) == 'mi chiamo chiara tu come ti chiami'
    assert remove('prova%$ https:/sito.com', nlp) == 'prova'

