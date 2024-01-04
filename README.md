# NLP_fundamentals (TBC)
## Essential tools and techniques from scratch

### 1. Text preprocessing and corpus preparation to faetures extraction
First of all, it’s required to know how to preprocess raw text data and extract features. You can face the total mess while scraping or assembling data. Thats why the NLP basics is essential for any DS/DA.
1. How to make raw data be ready for analysis?
    1. Put it onto a list or collection of a single sentences.
    2. Cleansing with `stopwords` and specific characters (and even numerical data) replacement.
    3. Some patterns excluding like emails, links, phones etc. 
    4. Converting to a lower case.
    5. Lemmatizing or/and stemming of provided corpus.
    6. Construct ngrams (optional).
    
   For this part you need to use some tools like regex manipulations and `NLTK` lib. The good idea is to setup the `TextBlob, Spacy, Scrapy` and others according to your goals. 
    
   Lets look at the “Tweets and User Engagement” dataset, provided by Kaggle ([link](https://www.kaggle.com/datasets/thedevastator/tweets-and-user-engagement/data)).
   In vast majority of cases we will go throw standard NLP routine writing algorithms and constructing from scratch.
