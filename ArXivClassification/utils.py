from __future__ import annotations


from itertools import cycle
from math import ceil
import re
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay, \
                            multilabel_confusion_matrix, auc, \
                            average_precision_score, \
                            precision_recall_curve, \
                            PrecisionRecallDisplay \
# Dependencies to import only for type checking.
if TYPE_CHECKING:
    from keybert import KeyBERT
    import sklearn
    from sklearn.pipeline import Pipeline
    import spacy


def is_string_series(s: pd.Series) -> bool:
    """
    Checks whether `s` series contains only strings.

    Parameters
    ----------
    s : pd.Series
        Series to be checked.
    """
    if isinstance(s.dtype, pd.StringDtype):
        # String series.
        return True

    elif s.dtype == 'object':
        # Object series --> must check each value.
        return all((v is None) or isinstance(v, str) for v in s)

    else:
        return False


def categories_as_lists(df: pd.DataFrame) -> None:
    """
    Ensures that the category column contains is a column
    of lists of strings.

    Parameters
    ----------
    df : pd.DataFrame
         Dataframe to be checked.
    """
    if(is_string_series(df["category"]) == True):
        df["category"] =  df["category"].map(eval)
      
    return


def categories_as_strings(df: pd.DataFrame) -> None:
    """
    Ensures that the category column is a column of strings.

    Parameters
    ----------
    df : pd.DataFrame
         Dataframe to be checked.
    """
    if(is_string_series(df["category"]) == False):
        df["category"] =  df["category"].map(str)
      
    return


def plot_df_counts(df: pd.DataFrame, col: str) -> dict:
    """
    Computes the occurrences of the lists in the `col` column
    of the `df` dataframe and plots histograms.

    Parameters
    ---------
    df : pd.DataFrame
         Dataframe that contains the column `col`.
    col : string
          Name of the column whose elements must be counted.

    Returns
    -------
    dict_counts : dictionary 
                  Its keys are the names contained in `col`'s lists 
                  and its values are their occurrences.
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

    # Plot the histogram.
    ax = df_counts.plot.bar(x=col, y='counts', 
                            color='crimson', figsize=(20,5))
    ax.grid(False)
    ax.set_xlabel('Category', fontsize=13)
    ax.set_ylabel('Counts', fontsize=13)
    ax.tick_params(labelsize=13)
  
    return dict_counts
    

def text_cleaner(text: str, nlp: spacy.lang.en.English) -> str:
    """
    After joining interrupted words and tokenizing the text, 
    lemmatize, remove bad words, special characters, punctuation
    and Spacy stopwords.

    Parameters
    ---------
    text : string
           Text to be cleaned.
    nlp : spacy.lang.en.English
          Spacy model.

    Returns
    -------
    clean_tokens : string
                   Cleaned text.
    """
    # Join interrupted words.
    text = text.replace("- ", "")  

    # Tokenize.
    tokens = nlp(text) # list of words

    # Lemmatize and transform to lowercase.
    #"-PRON-" is used as the lemma for all pronouns such as I, me, their ...
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" 
              else word.lower_ for word in tokens] 

    # Remove the tokens that contain any string in badlist.
    badlist = [".com", "http"] 
    clean_tokens = [word for word in tokens 
                    if not any(bad in word for bad in badlist)]
  
    # Remove stopwords.
    from spacy.lang.en.stop_words import STOP_WORDS
    clean_tokens = [word for word in clean_tokens 
                    if not word in STOP_WORDS] 

    # Keep only characters that are alphabet letters or spaces.
    clean_tokens = " ".join(c for c in clean_tokens 
                            if c.isalpha() or c.isspace()) 
      
    return clean_tokens


def plot_confusion_matrices(mat: np.ndarray, classes: np.ndarray) -> None:
    """
    Plot the confusion matrices normalizing on columns.

    Parameters
    ---------
    mat : np.ndarray
          Confusion matrices given by the classification.
    classes : np.ndarray
              Classes of classification.      
    """
    num_mat = len(mat) # number of confusion matrices we want to plot
    
    # Find the number of cells in the grid that will contain the num_mat subplots.
    num_rows = ceil(np.sqrt(num_mat))
    num_cols = ceil(num_mat/num_rows)
    num_cells = num_rows*num_cols
    rest = num_cells - num_mat

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12,12))
    axes = axes.ravel() # get flattened axes

    # Iterate over the cells.
    for i in range(num_cells):

        if i < num_mat:
            # Plot the matrix.
            disp = ConfusionMatrixDisplay(normalize(mat[i], axis=0, norm='l1'))
            disp.plot(ax=axes[i])
            disp.ax_.set_title(f"{classes[i]}")
            disp.ax_.set_xlabel("Predicted", fontsize=13)
            disp.ax_.set_ylabel("True", fontsize=13)

            # Only show x and y labels for the plots in the border.
            first_i_of_last_row = num_rows*num_cols - num_cols
            if i < first_i_of_last_row - rest:
                disp.ax_.set_xlabel('') # do not set the x label

            is_i_in_first_col = i%num_cols == 0
            if is_i_in_first_col == False:
                disp.ax_.set_ylabel('') # do not set the y label

            disp.im_.colorbar.remove() # remove it to put it after
            disp.ax_.grid(False)
            disp.ax_.tick_params(labelsize=13)

        else: # delete axes in excess
            fig.delaxes(axes[i])

    fig.colorbar(disp.im_, ax=axes)
    plt.show()
  
    return


def ROC(classes: np.ndarray, y_test: np.ndarray, y_score: np.ndarray) -> None:
    """
    Plot the ROC curves and compute their areas.

    Parameters
    ---------
    classes : np.ndarray
              Classes of classification.
    y_test : np.ndarray
             Test labels.
    y_score : np.ndarray
              Target scores.   
    """

    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class.
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
  
    # Sort the dictionary based on the area value.
    roc_auc_ord = dict(sorted(roc_auc.items(), key=lambda item: item[1]))

    # Take the sorted indices.
    indices = list(roc_auc_ord.keys())

    # Plot ROC curve.
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    # Compute micro-average ROC curve and ROC area.
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ax.plot(fpr["micro"], tpr["micro"], color='k',
            label='micro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["micro"]))

    # Do the same for each class.
    colors = ["dodgerblue", "gray", "crimson", "deeppink",
              "indigo", "turquoise", "orange"]
    lines = ['-', ':', '--']

    colorcyler = cycle(colors)
    linecycler = cycle(lines)

    num_colors = len(colors)

    linestyle = '-'
    for i, idx in enumerate(indices):

        if not i % num_colors:
            linestyle = next(linecycler)

        ax.plot(fpr[idx], tpr[idx], color=next(colorcyler), linestyle=linestyle,
                label='{0} (area = {1:0.2f})'.format(classes[idx], roc_auc[idx]))


    ax.set_xlim(-0.01,1.01)
    ax.set_ylim(-0.01,1.01)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.tick_params(labelsize=13)
    ax.legend(loc="lower right", fontsize='14', framealpha=0.5, ncol=1)
    plt.show()
  
    return


def PRC(classes: np.ndarray, y_test: np.ndarray, y_score: np.ndarray) -> None:
    """
    Plot the Precision-Recall curves. 
        
    Parameters
    ---------
    classes : np.ndarray
              The classes used in the classification.  
    y_test : np.ndarray
             Test labels.
    y_score : np.ndarray
              Target scores.
    """
    # Precision, recall for each class.
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i],
                                                       y_score[:, i])
      
    # Sort the dictionary based on the average_precision value.
    avg_precision_ord = dict(sorted(average_precision.items(), key=lambda item: item[1]))

    # Take the sorted indices.
    indices = list(avg_precision_ord.keys())
  
    # Micro-average of precision and recall.
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test,
                                                         y_score,
                                                         average="micro")
  
    # Plot f1-scores.
    _, ax = plt.subplots(figsize=(8, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)

    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("$f_1$={0:0.1f}".format(f_score),
                    xy=(0.83, y[45]+0.01),
                    fontsize=13)

    # Plot the micro-average.
    display = PrecisionRecallDisplay(recall=recall["micro"],
                                     precision=precision["micro"],
                                     average_precision=average_precision["micro"])

    display.plot(ax=ax, name="micro-average", color="black")

    # Plot a curve for each class.
    colors = ["dodgerblue", "gray", "crimson", "deeppink",
              "indigo", "turquoise", "orange"]
    lines = ['-', ':', '--']

    colorcyler = cycle(colors)
    linecycler = cycle(lines)
    
    num_colors = len(colors)
    
    linestyle = '-'
    for i, idx in enumerate(indices):
        display = PrecisionRecallDisplay(recall=recall[idx],
                                        precision=precision[idx],
                                        average_precision=average_precision[idx])
        if not i % num_colors:
            linestyle = next(linecycler)
            
        display.plot(ax=ax, color=next(colorcyler), name=f"{classes[idx]}", 
                     linestyle=linestyle)

    # Add the legend for the iso-f1 curves.
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])

    # Set the legend and the axes.
    ax.legend(handles=handles, labels=labels, loc="lower left", prop={'size': 13})
    ax.set_xlim(-0.01,1.01)
    ax.set_ylim(-0.01,1.01)
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.tick_params(labelsize=13)
    plt.show()



def extract_kws(text: str, kw_model: keybert._model.KeyBERT, seed: List[str], top_n: int) -> List[str]:
    """
    Extract a list of `top_n` keywords for the input text
    using some seed-keywords given by `seed`.

    Parameters
    ---------
    text : string
           Text from which to extract keywords.
    kw_model : keybert._model.KeyBERT
               KeyBERT model.
    seed : list of strings
           Seed keywords that might guide the extraction of keywords.
    top_n : int
            Number of keywords to extract

    Returns
    -------
    keywords: list of strings
              List of the top_n extracted keywords.
    """
  
    max_n_grams = 1
    if seed == ['']: # if there are no words to use as seeds
        seed = None # switch off seed_keywords parameter below
    data = kw_model.extract_keywords(docs=text,
                                     keyphrase_ngram_range=(1, max_n_grams),
                                     seed_keywords = seed,
                                     stop_words='english',
                                     use_mmr=True,
                                     top_n=top_n)
    keywords = list(list(zip(*data))[0])
  
    return keywords
