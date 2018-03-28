**Identifying Superheroes from Product Images** 

**Description**

While using machine learning to perform image recognition is currently one of the most popular use cases, in some cases, the existing large-scale models are too broad to be effective for specific business use cases. In this contest we will use a data driven approach to identify the “superheroes” in an image (fashion product images). 
Objective

The objective is as follows: Identify the “superhero” from each image from a list of 12 possible superheroes:

    Antman
    Aquaman
    Avengers
    Batman
    Black Panther
    Captain America
    Catwoman
    Ghostrider
    Hulk 
    Ironman
    Spiderman
    Superman

The contest can be located at [CrowdAnalytix Site](https://crowdanalytix.com/contests/identifying-superheroes-from-product-images)

**Data**

Dataset consists of product images like t-shirts, bags etc. with superhero graphics. The master data set has been split into training and test data sets for the contest tasks. Participants will use the data sets provided to develop and test their models for prediction. You can download the training and test data from data tab. The data distribution across training and test data sets are as follows:

    Training: 5433 images across 12 categories 
    Test: 3375 images



**Evaluation Criterion:**

    Overall Accuracy i.e. [Correct Prediction / n]

**Best Model**
By training an InceptionV3 model through Transfer Learning was able to achive a Private Leaderboard accuracy score of 81% which translated to 76% in the Private Leaderboard. The code can be 
located at 

- [x] [script.py ](https://github.com/santanupattanayak1/ML_DS_Catalog-/blob/master/Superheroes%20Crowd%20Analytics/script.py)










