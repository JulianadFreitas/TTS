
# About

This folder is organized as follows:

- Inside the `estimators_analysis` folder you will find all the files containing the performances of the classification algorithms we tested for our dataset (the complete list of algorithms is available in our paper). Inside it, you will also find the best parameters according to GridSearch.
- In the `classification_analysis` folder your will find the later stages of the classification model, where LinearSVC was selected and we were testing different settings for it. Different models, confusing matrices and learning curves are available based on the tests we made (e.g. with or withouth the heuristic features).
- Inside the 'feature_analysis' folder you will find the files showing the best features per class. This analysis was done after we trained our final model, and the process to find the best features is described in our paper (You may also look at our code inside the `scripts/classifier` folder).
- The `images` folder contains the charts generated using Excel. You will also a customizable version of each chart in PowerPoint.
- Inside the `survey` folder you will find the data from participants who responded to our online questionnaire. 
- The spreadsheet `documentation_analysis.xlsx` file contains the overall analysis we made for this paper based on the values we gathared through the entire method. Each worksheet is supposed to be as simple as possible, so I hope you don't struggle with it.
