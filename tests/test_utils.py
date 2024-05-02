import re

import en_core_web_md
import spacy

from arXiv_classification.utils import remove

def test_remove():
    """Unit test for the remove() function.
    """
    nlp = spacy.load("en_core_web_md")
    assert remove('hi_3 !? Hi, 374189 h5i \math{hi} hi%& https:/website.com GitHub.com re-move\n', nlp) == 'hi'
