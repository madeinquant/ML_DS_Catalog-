**Description**

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

The Conversation AI team, a research initiative founded by Jigsaw and Google (both a part of Alphabet) are working on tools to help improve online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful or otherwise likely to make someone leave a discussion). So far they’ve built a range of publicly available models served through the Perspective API, including toxicity. But the current models still make errors, and they don’t allow users to select which types of toxicity they’re interested in finding (e.g. some platforms may be fine with profanity, but not with other types of toxic content).

In this competition, you’re challenged to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. You’ll be using a dataset of comments from Wikipedia’s talk page edits. Improvements to the current model will hopefully help online discussion become more productive and respectful.

[Competition link](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

**Data Description**

Data Description

You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

    toxic
    severe_toxic
    obscene
    threat
    insult
    identity_hate

You must create a model which predicts a probability of each type of toxicity for each comment.

[Data Link](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)


**Scripts Description** 


- [x] [Bi Drectional GRU with Fast text](https://github.com/santanupattanayak1/ML_DS_Catalog-/blob/master/Kaggle%20Toxic%20Comment%20Classification%20Challenge/Scripts%20/exp_gru_clean.py)

- [x] [Bi Directional GRU followed by Convolution](https://github.com/santanupattanayak1/ML_DS_Catalog-/blob/master/Kaggle%20Toxic%20Comment%20Classification%20Challenge/Scripts%20/cnnlstm.py)

- [x] [Lightgbm using Latent Dirchlet ALlocation, Local Sensitive Hashing, latent Semantic Analysis, Sentiment Polarityt based features](https://github.com/santanupattanayak1/ML_DS_Catalog-/blob/master/Kaggle%20Toxic%20Comment%20Classification%20Challenge/Scripts%20/LDA%20LSA%20LSH%20features%20based%20Lightgbm.py)

- [x] [Lightgbm on Char vectors](https://github.com/santanupattanayak1/ML_DS_Catalog-/blob/master/Kaggle%20Toxic%20Comment%20Classification%20Challenge/Scripts%20/lightgbm.py)

- [x] [Factorization Machines FM_FTRL on Char vectors](https://github.com/santanupattanayak1/ML_DS_Catalog-/blob/master/Kaggle%20Toxic%20Comment%20Classification%20Challenge/Scripts%20/char_vec_ftrl.py)

- [x] [Logistic Regression on Char vectors](https://github.com/santanupattanayak1/ML_DS_Catalog-/blob/master/Kaggle%20Toxic%20Comment%20Classification%20Challenge/Scripts%20/logistic%20with%20char%20 vector.py)


**Results**

*With the emsemble of results from the above models was able to achieve Private Leaderboard score of 0.9849 AUC.*


