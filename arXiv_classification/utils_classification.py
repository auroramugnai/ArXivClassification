import numpy as np
import re
import spacy
import matplotlib.pyplot as plt
from math import ceil
from sklearn.preprocessing import normalize
from sklearn.metrics import  roc_curve, ConfusionMatrixDisplay, multilabel_confusion_matrix, auc


def run_SVM_pipeline_one(pipeline, X_train, X_test, y_train, y_test):
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
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return y_pred


def run_SVM_pipeline_multi(pipeline, X_train, X_test, y_train, y_test):
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
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mat = multilabel_confusion_matrix(y_test, y_pred) # confusion matrices
    return y_pred, mat


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




def print_confusion_matrices(mat, classes):
    """
    Represent the confusion matrices.

    Arguments
    ---------
       mat: confusion matrices given by the classification
    """
    num_mat = len(mat) # number of confusion matrices we want to plot
    # find the number of cells in the grid that will contain the num_mat subplots
    num_rows = ceil(np.sqrt(num_mat))
    num_cols = ceil(num_mat/num_rows)
    num_cells = num_rows*num_cols
    rest = num_cells - num_mat

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,20))
    axes = axes.ravel() # get flattened axes

    # Iterate over the cells
    for i in range(num_cells):

        if i < num_mat:
            # Plot the matrix.
            disp = ConfusionMatrixDisplay(normalize(mat[i], axis=0, norm='l1'))
            disp.plot(ax=axes[i])
            disp.ax_.set_title(f'{classes[i]}')

            # Only show x and y labels for the plots in the border
            first_i_of_last_row = num_rows*num_cols - num_cols
            if i < first_i_of_last_row - rest:
                disp.ax_.set_xlabel('') # Do not set the x label

            is_i_in_first_col = i%num_cols == 0
            if is_i_in_first_col == False:
                disp.ax_.set_ylabel('') # Do not set the y label

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


    # Sort the dictionary based on the area value
    roc_auc_ord = dict(sorted(roc_auc.items(), key=lambda item: item[1]))

    # Take the sorted indices
    indici = list(roc_auc_ord.keys())

    ##### Plot ROC curve #####
    fig = plt.figure(figsize=(10,10))
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
    plt.legend(loc="lower right", fontsize='6')

    return