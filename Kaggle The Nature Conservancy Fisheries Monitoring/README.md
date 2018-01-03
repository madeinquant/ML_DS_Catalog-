
**Description**

Nearly half of the world depends on seafood for their main source of protein. In the Western and Central Pacific, where 60% of the world’s tuna is caught, illegal, unreported, and unregulated fishing practices are threatening marine ecosystems, global seafood supplies and local livelihoods. The Nature Conservancy is working with local, regional and global partners to preserve this fishery for the future.

The Nature Conservancy Competition

Currently, the Conservancy is looking to the future by using cameras to dramatically scale the monitoring of fishing activities to fill critical science and compliance monitoring data gaps. Although these electronic monitoring systems work well and are ready for wider deployment, the amount of raw data produced is cumbersome and expensive to process manually.

The Conservancy is inviting the Kaggle community to develop algorithms to automatically detect and classify species of tunas, sharks and more that fishing boats catch, which will accelerate the video review process. Faster review and more reliable data will enable countries to reallocate human capital to management and enforcement activities which will have a positive impact on conservation and our planet.

Machine learning has the ability to transform what we know about our oceans and how we manage them. You can be part of the solution.


In this competition, The Nature Conservancy asks you to help them detect which species of fish appears on a fishing boat, based on images captured from boat cameras of various angles.  

Your goal is to predict the likelihood of fish species in each picture.

Eight target categories are available in this dataset: Albacore tuna, Bigeye tuna, Yellowfin tuna, Mahi Mahi, Opah, Sharks, Other (meaning that there are fish present but not in the above categories), and No Fish (meaning that no fish is in the picture). Each image has only one fish category, except that there are sometimes very small fish in the pictures that are used as bait. 

The dataset was compiled by The Nature Conservancy in partnership with Satlink, Archipelago Marine Research, the Pacific Community, the Solomon Islands Ministry of Fisheries and Marine Resources, the Australia Fisheries Management Authority, and the governments of New Caledonia and Palau.



**Approach**

- [x] Trained 5 models based on 5 fold cross validation method 
- [x] Each of the models has the same customized architecture 



**CNN Architecture**

Layer name | Desc
-----------|-----------------------------------------------
Conv | 3x3 Filter,64 feature maps,Activation ReLU
Conv | 3x3 Filter,64 feature maps,Activation ReLU
Pool | 2x2 MaxPooling 
Conv | 3x3 Filter,128 feature maps,Activation ReLU
Conv | 3x3 Filter,128 feature maps,Activation ReLU
Pool | 2x2 MaxPooling
Conv | 3x3 Filter,256 feature maps,Activation ReLU
Conv | 3x3 Filter,256 feature maps,Activation ReLU
Pool | 2x2 MaxPooling
Conv | 3x3 Filter,512 feature maps,Activation ReLU
Conv | 3x3 Filter,512 feature maps,Activation ReLU
Pool | 2x2 MaxPooling
Conv | 3x3 Filter,512 feature maps,Activation ReLU
Conv | 3x3 Filter,512 feature maps,Activation ReLU
Pool | 2x2 MaxPooling
Flatten| Flatten 
FC Layer|256 Dense units, Fully Connected layer, Activation ReLU
Dropout|0.5
FC Layer|256 Dense units, Fully Connected layer, Activation ReLU
Dropout|0.5
Output layer|8 units Softmax 

- [x] Take the average prediction of the 5 models 

*Kaggle link to the Competition*
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

*Kaggle link to the dataset*
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data

*Kaggle Profile*
https://www.kaggle.com/santanuds


 
    


