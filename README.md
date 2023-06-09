#news-topic-modeling
==============================

This project involves using the News API to scrape news articles and applying topic modeling techniques to identify the main topics in the articles. The resulting topics are then visualized using dash(Plotly), a data visualization framework built on top of Flask
* The project aims to provide insights into the main topics covered in the news articles and to help users quickly identify the most relevant articles based on their interests. The following steps are involved in the project:\
1- Scraping news articles using the News API\
2- Preprocessing the text data by removing stop words, stemming, and lemmatizing\
3- Applying topic modeling techniques such as Latent Dirichlet Allocation (LDA) to identify the main topics in the articles\
4- Visualizing the topics using Plotly see [dashboard](https://github.com/mustaphaamine1/News-Topic-Modeling/blob/main/Dash.pdf) \

Project Organization
------------

    ├── LICENSE
    ├── README.md 
    ├── Dash.pdf            <- The final dash app. 
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable datasets.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized LDA models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         installed with `pip install -r requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │  
    │   ├── data           <- Scripts to generate and download data
    │   │   ├── Latest_news_dataset.py
    │   │   └── Search_news_dataset.py
    │   │  
    │   ├── pre-processing      <- Scripts to clean the raw data
    │   │   └── clean_data.py
    │   │  
    │   ├── models         <- Scripts to create the LDA models
    │   │   └── topic_modeling.py
    │   │  
    │   ├── visualization  <- Script to create the dashboard
    │       └── app.py
    └──


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
