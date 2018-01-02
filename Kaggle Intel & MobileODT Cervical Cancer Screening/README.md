
**Description**


Cervical cancer is so easy to prevent if caught in its pre-cancerous stage that every woman should have access to effective, life-saving treatment no matter where they live. Today, women worldwide in low-resource settings are benefiting from programs where cancer is identified and treated in a single visit. However, due in part to lacking expertise in the field, one of the greatest challenges of these cervical cancer screen and treat programs is determining the appropriate method of treatment which can vary depending on patients’ physiological differences.

Intel MobileODT

Especially in rural parts of the world, many women at high risk for cervical cancer are receiving treatment that will not work for them due to the position of their cervix. This is a tragedy: health providers are able to identify high risk patients, but may not have the skills to reliably discern which treatment which will prevent cancer in these women. Even worse, applying the wrong treatment has a high cost. A treatment which works effectively for one woman may obscure future cancerous growth in another woman, greatly increasing health risks.

Currently, MobileODT offers a Quality Assurance workflow to support remote supervision which helps healthcare providers make better treatment decisions in rural settings. However, their workflow would be greatly improved given the ability to make real-time determinations about patients’ treatment eligibility based on cervix type.

In this competition, Intel is partnering with MobileODT to challenge Kagglers to develop an algorithm which accurately identifies a woman’s cervix type based on images. Doing so will prevent ineffectual treatments and allow healthcare providers to give proper referral for cases that require more advanced treatment.



**Approach**

- [x] Trained 5 models based on 5 fold cross validation method 
- [x] Each of the models has the same customized architecture 

    *CNN Architecture*
3x3 Conv , 64 feature maps,Activation ReLU
----------------------------------------------------------- 
         
 3x3 Conv , 64 feature maps,Activation ReLU
---------------------------------------------------------- 
    2x2 MaxPooling 
---------------------------------------------------------- 
 3x3 Conv , 128 feature maps,Activation ReLU
---------------------------------------------------------- 
 3x3 Conv , 128 feature maps,Activation ReLU
---------------------------------------------------------- 
 2x2 MaxPooling 
---------------------------------------------------------- 
 3x3 Conv , 256 feature maps,Activation ReLU
---------------------------------------------------------- 
 3x3 Conv , 256 feature maps,Activation ReLU
---------------------------------------------------------- 
 3x3 Conv , 256 feature maps,Activation ReLU
---------------------------------------------------------- 
 2x2 MaxPooling 
---------------------------------------------------------- 
 3x3 Conv , 512 feature maps,Activation ReLU
---------------------------------------------------------- 
 3x3 Conv , 512 feature maps,Activation ReLU
---------------------------------------------------------- 
 2x2 MaxPooling 
---------------------------------------------------------- 
     Flatten
---------------------------------------------------------- 
 512 Dense units, Fully Connected layer, Activation ReLU
---------------------------------------------------------- 
 Dropout 0.5
---------------------------------------------------------- 
 512 Dense units, Fully Connected layer, Activation ReLU
---------------------------------------------------------- 
 Dropout 0.5
---------------------------------------------------------- 
 Dense 3, Output layer
---------------------------------------------------------- 

- [x] Take the average prediction of the 5 models 

*Kaggle link to the Competition*
https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening

*Kaggle link to the dataset*
https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data

*Kaggle Profile*
https://www.kaggle.com/santanuds


 
    


