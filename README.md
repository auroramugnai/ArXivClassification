An algorithm is developed to classify scientific articles based on their topic. For this purpose a fraction of the arXiv dataset available at https://www.kaggle.com/datasets/Cornell-University/arxiv is used.

ArXivClassification performs:

1) Supervised classification:

	• where the labels are the articles'lists of categories and the features are their abstracts and titles;

	• where the labels are the articles'lists of categories and the features are their lists of keywords, extracted from abstract and title;

	• where the labels are the articles'keywords (one per article) and the features are their abstracts and titles (narrowing down to a single macro-category).

2) Unsupervised classification:

	• Using only abstracts and titles of the articles.
