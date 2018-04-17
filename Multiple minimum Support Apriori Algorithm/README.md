**Multiple Minimum Support Apriori Algorithm**

*This is a Python implementation of Multiple Minimum Support Apriori Algorithm. The inputs required to the Algorithm are as below 

- Input transaction dataset with transaction ids 
- Global Minimum Support ( probability)   global_min_supp
- beta, The minimum support threshold for an item is the max(global_min_supp,beta*support)
- Support difference threshold phi_thresh, if in an itemset the support difference between any pair is greater than phi_thresh ther itemset is not selected.
- Run with or without item constraint item_contraint (Y/N) Y - *Run with item constraint*   N  - *Run without item constraint*. The list of items that cant be toghether should be mentioned in 
*items_cannot*  field while the items that must be in present in the itemsets should be mentioned in *items_must* field.


*Note* - The file records may need to be split differently in different operating system because of the different record end characters. The code is for windows where the final terminating character is 
/n. 







