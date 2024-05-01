import re
from math import ceil

import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import  roc_curve, ConfusionMatrixDisplay, multilabel_confusion_matrix, auc

def is_string_series(s: pd.Series) -> bool:
    """Checks whether s series contains only strings.
    """
    if isinstance(s.dtype, pd.StringDtype):
        # The series was explicitly created as a string series (Pandas>=1.0.0).
        return True

    elif s.dtype == 'object':
        # Object series, check each value.
        return all((v is None) or isinstance(v, str) for v in s)

    else:
        return False


def categories_as_lists(df: pd.DataFrame) -> None:
    """Ensures that the category column contains lists of strings.
    """
    if(is_string_series(df["category"]) == True):
        df["category"] =  df["category"].map(eval)
    return


def categories_as_strings(df: pd.DataFrame) -> None:
    """Ensures that the category column contains strings.
    """
    if(is_string_series(df["category"]) == False):
        df["category"] =  df["category"].map(str)
    return


def plot_df_counts(df: pd.DataFrame, col: str) -> dict:
    """
    Computes the occurrences of the lists in the col column
    of the df dataframe and plots histograms. Everything is
    repeated for the exploded dataframe.

    Arguments
    ---------
       df:  dataframe
       col: string, name of the column whose elements must be counted

    Returns
    -------
       dict_counts: a dictionary whose keys are the names contained
                    in col's lists and whose values are their occurrences
    """

    # Before grouping check if we have strings.
    categories_as_strings(df)

    # Grouping by the 'col' column and counting the occurrences.
    df_counts = df[[col]].groupby([col])[col].count()
    df_counts = df_counts.reset_index(name='counts')

    # Sorting in descending order.
    df_counts = df_counts.sort_values(['counts'], ascending=False)
    df_counts = df_counts.reset_index(drop=True)

    # Creating a dictionary.
    names = df_counts[col].tolist()
    counts = df_counts['counts'].tolist()
    dict_counts = dict([(v, c) for v, c in zip(names, counts)])

    # Plot.
    df_counts.plot.bar(x=col, y='counts', 
                       color='r', figsize=(20,5))
    return dict_counts


def run_model_one(pipeline, X_train, X_test, y_train, y_test):
    """
    Execute the fit and prediction for the classification using the
    defined SVM_Pipeline, that vectorize and classify the data.

    Arguments
    ---------
       pipeline: defined Pipeline
       X_train: train features
       X_test: test features
       y_train: numpy.ndarray, train labels
       y_test: numpy.ndarray, test labels

    Returns
    -------
       y_pred: predictions of the test data
    """
    # Fit of the train data using the pipeline.
    pipeline.fit(X_train, y_train)
    # Prediction on the test data.
    y_pred = pipeline.predict(X_test)
    return y_pred


def run_model_multi(pipeline, X_train, X_test, y_train, y_test):
    """
    Execute the fit and prediction for the classification using the
    defined SVM_Pipeline, that vectorize and classify the data.
    Compute the confusion matrices.

    Arguments
    ---------
       pipeline: defined Pipeline
       X_train: train features
       X_test: test features
       y_train: numpy.ndarray, train labels
       y_test: numpy.ndarray, test labels

    Returns
    -------
       y_pred: predictions of the test data
       mat: confusion matrices
    """
    # Fit of the train data using the pipeline.
    pipeline.fit(X_train, y_train)
    # Prediction on the test data.
    y_pred = pipeline.predict(X_test)
    # Compute the confusion matrices.
    mat = multilabel_confusion_matrix(y_test, y_pred)
    return y_pred, mat


def remove(text, nlp):
    """
    After tokenizing the text, remove punctuation and other characters.

    Arguments
    ---------
       text: str, text to be processed
       nlp: Spacy model

    Returns
    -------
       filtered_text: str, processed text
    """
    # Apply the natural language Spacy model to the input text.
    tokens = nlp(text)
    # Transform to lowercase and then split the text.
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens] 
    # Remove the tokens that contains those strings
    B = ["github.com", "http"] 
    filtered_text = [word for word in tokens if not any(bad in word for bad in B)] # filtered_text must be a list of words
    # Remove the numeric characters.
    filtered_text = " ".join(c for c in filtered_text if c.isalpha() or c.isspace()) 
    filtered_text = re.sub('(?:\s)http[^, ]*', '', filtered_text) # remove the words that begin with http (filtered_text must be a string)
    filtered_text = filtered_text.replace("- ", "")  # join the words that are hyphenated
    return filtered_text


def print_confusion_matrices(mat, classes):
    """
    Plot the confusion matrices normalizing on columns.

    Arguments
    ---------
       mat: confusion matrices given by the classification
    """
    num_mat = len(mat) # number of confusion matrices we want to plot
    
    # Find the number of cells in the grid that will contain the num_mat subplots.
    num_rows = ceil(np.sqrt(num_mat))
    num_cols = ceil(num_mat/num_rows)
    num_cells = num_rows*num_cols
    rest = num_cells - num_mat

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,20))
    axes = axes.ravel() # get flattened axes

    # Iterate over the cells.
    for i in range(num_cells):

        if i < num_mat:
            # Plot the matrix.
            disp = ConfusionMatrixDisplay(normalize(mat[i], axis=0, norm='l1'))
            disp.plot(ax=axes[i])
            disp.ax_.set_title(f'{classes[i]}')

            # Only show x and y labels for the plots in the border.
            first_i_of_last_row = num_rows*num_cols - num_cols
            if i < first_i_of_last_row - rest:
                disp.ax_.set_xlabel('') # do not set the x label

            is_i_in_first_col = i%num_cols == 0
            if is_i_in_first_col == False:
                disp.ax_.set_ylabel('') # do not set the y label

            disp.im_.colorbar.remove() # remove it to put it after

        else: # delete axes in excess
            fig.delaxes(axes[i])


    plt.subplots_adjust(wspace=0.15, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)
    return


def ROC(classes, y_test, y_score):
    """
    Plot the ROC curves and compute their areas.

    Arguments
    ---------
       classes: numpy.ndarray with all the possible categories
       X_train: train features
       X_test: test features
       y_train: train labels
    """

    n_classes = len(classes)

    ##### Compute ROC curve and ROC area for each class #####
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # Sort the dictionary based on the area value.
    roc_auc_ord = dict(sorted(roc_auc.items(), key=lambda item: item[1]))

    # Take the sorted indices.
    indici = list(roc_auc_ord.keys())

    ##### Plot ROC curve #####
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle('color', [cm(1.*i/n_classes) for i in range(n_classes)])

    for i in indici:
        ax.plot(fpr[i], tpr[i], label='{0} (area = {1:0.2f})'
                                      ''.format(classes[i], roc_auc[i]))


    ###### Compute micro-average ROC curve and ROC area #####
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ax.plot(fpr["micro"], tpr["micro"], color='k',
            label='micro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["micro"]))

    ax.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curves')
    plt.legend(loc="lower right", fontsize='6', aplha=0.5)
    return


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
