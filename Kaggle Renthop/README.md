Finding the perfect place to call your new home should be more than browsing through endless listings. RentHop makes apartment search smarter by using data to sort rental listings by quality. But while looking for the perfect apartment is difficult enough, structuring and making sense of all available real estate data programmatically is even harder. Two Sigma and RentHop, a portfolio company of Two Sigma Ventures, invite Kagglers to unleash their creative engines to uncover business value in this unique recruiting competition.


Two Sigma invites you to apply your talents in this recruiting competition featuring rental listing data from RentHop. Kagglers will predict the number of inquiries a new listing receives based on the listing’s creation date and other features. Doing so will help RentHop better handle fraud control, identify potential listing quality issues, and allow owners and agents to better understand renters’ needs and preferences.


Two Sigma has been at the forefront of applying technology and data science to financial forecasts. While their pioneering advances in big data, AI, and machine learning in the financial world have been pushing the industry forward, as with all other scientific progress, they are driven to make continual progress. This challenge is an opportunity for competitors to gain a sneak peek into Two Sigma's data science work outside of finance.

In this competition, you will predict how popular an apartment rental listing is based on the listing content like text description, photos, number of bedrooms, price, etc. The data comes from renthop.com, an apartment listing website. These apartments are located in New York City.

The target variable, interest_level, is defined by the number of inquiries a listing has in the duration that the listing was live on the site. 
File descriptions

    train.json - the training set
    test.json - the test set
    sample_submission.csv - a sample submission file in the correct format
    images_sample.zip - listing images organized by listing_id (a sample of 100 listings)
    Kaggle-renthop.7z - (optional) listing images organized by listing_id. Total size: 78.5GB compressed. Distributed by BitTorrent (Kaggle-renthop.torrent). 

Data fields

    bathrooms: number of bathrooms
    bedrooms: number of bathrooms
    building_id
    created
    description
    display_address
    features: a list of features about this apartment
    latitude
    listing_id
    longitude
    manager_id
    photos: a list of photo links. You are welcome to download the pictures yourselves from renthop's site, but they are the same as imgs.zip. 
    price: in USD
    street_address
    interest_level: this is the target variable. It has 3 categories: 'high', 'medium', 'low'





