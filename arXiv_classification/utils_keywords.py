def remove(text, nlp):
    """
    After tokenizing the text, remove punctuation and other characters.

    Arguments
    ---------
       text: str, text to be processed
       nlp: spacy model

    Returns
    -------
       filtered_text: str, processed text
    """
    tokens = nlp(text)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens] # transform to lowercase and then split the text
    B = ["github.com", "http"] # remove the tokens that contains those strings
    filtered_text = [word for word in tokens if not any(bad in word for bad in B)] # filtered_text must be a list of words
    filtered_text = " ".join(c for c in filtered_text if c.isalpha() or c.isspace()) # remove the numeric charcaters
    filtered_text = re.sub('(?:\s)http[^, ]*', '', filtered_text) # remove the words that begin with http (filtered_text must be a string)
    filtered_text = filtered_text.replace("- ", "")  # join the words that are hyphenated

    return filtered_text

def extract_kws(TEXT, kw_model, seed, max_n_grams=1):
    """
    Extract a list of 4 keywords for each input text.

    Arguments
    ---------
       TEXT: text from which to extract keywords
       kw_model: KeyBERT model
       seed: seed keywords that might guide the extraction of keywords
             by steering the similarities towards the seeded keywords.
       max_n_grams: lenght of the keyword

    Returns
    -------
       keywords: list of extracted keywords for each text
    """
    data = kw_model.extract_keywords(docs=TEXT,
                                      keyphrase_ngram_range=(1,max_n_grams),
                                      seed_keywords = seed,
                                      stop_words='english',
                                      use_mmr=True,
                                      top_n=4) # number of keywords
    keywords = list(list(zip(*data))[0])
    return keywords
