# Predicting Risk of Depression in Youth
<p align="center">
  <img src="https://github.com/nlt-python/youth_behavioral_risk/blob/master/images/smiley.png">
</p>


## Motivation and Objective:

This project used machine learning models to predict youth behavior, specifically risk of depression as a function of various youth characteristics and other behavior. Models were derived from responses provided to a cross-sectional CDC survey on Youth Risk Behavior (YRBS). An interactive website was created using the best performing model to provide awareness on the topic to youth and their parents.



## Data Preparation & EDA:

The CDC's biennial survey on youth risk behavior began in 1991 on the national, state and district levels. This project only uses the national survey responses from youths attending school and ranging in age from 12 to 18 years old during 2017. The data was obtained from the [CDC](https://www.cdc.gov/healthyyouth/data/yrbs/data.htm) as a Microsoft Access file, converted to a csv file and then pandas dataframe for data cleaning and preparation.

The more than 100 questions are categorized into:

- Demographics
- Height and Weight

and the following areas of risk:

- Unintentional Injuries and Violence
- Tobacco Use
- Electronic Vapor Product Use
- Alcohol and Other Drug Use
- Sexual Behaviors that Contribute to Unintended Pregnancy and Sexually 
  Transmitted Diseases (STDs), Including Human Immunodeficiency Virus (HIV)
  Infection
- Weight Management
- Dietary Behaviors
- Physical Inactivity
- HIV
- Other Topics


The target is the binary repsonse to question:

“During the past 12 months, did you ever feel so sad or hopeless almost every day for two weeks or more in a row that you stopped doing some usual activities?”

This response was codified such that Yes is 1 and No is 0.


Each category in the CDC survey questions included redundancy to assess consistency among student responses. Redundant questions were removed to minimize collinearity in the features. Additional steps to clean the data are addressed in the [clean_data helper function](src/helpers.py) 



Simple correlations and visualizations were made to gain a cursory understanding of the data.

<p align="center">
  <img src="images/gender_race.png">
</p>


There were slightly more female than male students (approximately 12000 to 10000, respectively) and there are more youths identifying as 'Hispanic/Latino' than 'Black' and 'White' identifying students combined. While students were equally distributed amongst the 9th, 10th, 11th and 12th grades, most of the students were between 15 to 17 years of age.


<p align="center">
  <img src="images/age_grade.png">
</p>


Pairwise correlation of the dataset shows there may be a correlation between the group of features in the lower right corner of the heatmap. These features include being cyber bullied, using electronic vapor products and the age in which the youth first used marijuana.

*** PLACEHOLDER ***
<p align="center">
  <img src="images/corr_heat_map.png">
</p>

## Models:

A classifier was built using machine learning techniques and evaluated according to its precision, recall, receiver operating characteristics and a confusion matrix. 

Models to predict the probability a youth 'ever considered suicide' in the last year were made using logistic regression, k-nearest neighbors, naive bayes, random forest classification, gradient boost classification and ada boost classification. About 14 % of the respondents answered Yes to target variable, 'ever considered suicide', suggesting class imbalance exists in the dataset.



## TAKEAWAYS:

*** PLACEHOLDER *** ADD METRICS TABLE

DISCUSS FEATURE ENGINEERING, TECHNIQUES TO CONSIDER CLASS IMBALANCE

COMPARE MODELS

ADD IMPROVED METRICS TABLE USING BEST MODEL. DISCUSS ADJUSTMENTS TO MAKE IMPROVEMENTS. 

DISCUSS FEATURE IMPORTANCE and DISCUSS ITS INCORPORATION INTO WEBSITE

FINISH MAKING WEBSITE
ADD LINK TO WEBSITE!


## Featured Notebooks/Analysis Files:
