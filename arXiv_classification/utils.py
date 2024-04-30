import pandas as pd

def is_string_series(s: pd.Series) -> bool:
    """Checks whether s series contains only strings.
    """
    if isinstance(s.dtype, pd.StringDtype):
        # The series was explicitly created as a string series (Pandas>=1.0.0)
        return True

    elif s.dtype == 'object':
        # Object series, check each value
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

    # before grouping check if we have strings
    categories_as_strings(df)

    # grouping by the 'col' column and counting the occurrences.
    df_counts = df[[col]].groupby([col])[col].count()
    df_counts = df_counts.reset_index(name='counts')

    # sorting in descending order
    df_counts = df_counts.sort_values(['counts'], ascending=False)
    df_counts = df_counts.reset_index(drop=True)

    # creating a dictionary
    names = df_counts[col].tolist()
    counts = df_counts['counts'].tolist()
    dict_counts = dict([(v, c) for v, c in zip(names, counts)])

    # plot
    df_counts.plot.bar(x=col, y='counts', title=f"df-{col}", figsize=(20,5))

    return dict_counts
