# Predicting Customer Churn In The Start-Up Industry Using Multiple Machine Learning Algorithms

## Abstract

Costumer churn prediction, if accurate, can allow companies to greatly increase their revenue, as it is more cost effective to retain current customers than to attempt to obtain new customers. This consumer analysis tactic has gained in popularity over recent years, thus so has the use of machine learning algorithms to produce the prediction. Customer churn prediction literature suggests that certain algorithms are found to be extremely effective at aiding companies in this analysis. Thus, this study will attempt to detect the factors leading to customer churn at a small start-up. This will be achieved by executing 5 different machine learning algorithms where by they will all generate costumer churn rates for said company. The algorithms which will be applied are logistic regression, decision trees, random forest, adaboost, and support vector machines. Each algorithm will be investigated through the use of multiple classification algorithm metrics, including area under the curve analysis, accuracy, sensitivity, and F1 score. The analysis of these metrics will explore goodness of fit and determine the overall effectiveness of each model. The use of these metrics will allow models across different algorithms to be compared. The result of this study discovered that tree-based models were found to be the most effective, with a decision tree model producing the strongest results in determining customer churn rates at this small start-up. Ultimately, customer churn analysis can be extremely difficult to predict and this study is limited by a small data size and a lack of information regarding the company being analyzed. Customer churn prediction is a fast-growing field and more research in this area is needed. 

## Introduction

Revenue, growth, income and efficiency are various words that encapsulate the ultimate goal of every profit generating business. However, these words are often followed by questions of how. How will this revenue be generated? How will maximum gains be ensured and how will the business grow? The most basic answer to all these questions is to build a strong customer base. However, although bringing in new costumers is vital to the initial success of a business, a business cannot be expected to operate at its maximum capabilities when only relying on new customer acquisition, retaining its current customers must also be a priority (Vafeiadis et al., 2015). The act of calculating the rates of a customer leaving or cancelling their subscription to a business can be referred to as customer churn rate (Gold, 2020). Analyzing and optimizing customer churn rates are essential for any successful business as customer acquisition is far more expensive than customer retention (Vafeiadis et al., 2015). Thus, in order to optimize an organization, data analysis must be used to determine the underlying patterns and influences which have the largest potential to cause a customer to churn from a business. Once the population with the highest churn potential is identified, an organization is then able to target and adapt to these specific customers wants or needs before they lose them. This process is achieved through the use of various machine learning techniques which work to discover and detect patterns in costumer behaviour which can then be used to predict costumer churn. In order to further examine this phenomena, a case study has been produced which is focused on a small start-up company that desires to explore its own customer churn rate for the purpose of company improvement.

The purpose of this study is to attain the strongest predictor model of customer churn rates for the business in question which would ultimately lead to it being able to cut costs and increase customer retention. To find the strongest model, five different machine learning techniques will be applied and analyzed using this organization’s specific data. The use of various models is necessary because it is not possible to predict which model will be the most well-matched for this specific data. Accordingly, the five machine learning techniques being used are logistic regression, support vector machines (SVM), random forest, decision trees, and adaboost. All techniques used are of industry standard and are generally believed to provide the strongest predictions of customer churn. This study also desires to contribute to the larger industry understanding of churn rate modelling techniques and specifically to the understanding of predictors of customer churn in the start-up sector. This study is limited by the scope of understanding of the company in question, as little company information is provided, thus, this creates a lack of context for the variables being manipulated.

The study will be structured as follows, the literature review section will provide justification for the techniques used and will review previous studies which provide context for this analysis. Following, the methodology section will provide an explanation of the data set, the data analysis techniques selected, and the platforms used. Subsequently, the results section will present the findings of the five models and will use various metrics to allow for a comparison between them. It will additionally address the validity, errors, and accuracy of each model. Finally, the discussion section will consider the final findings of the research and will seek to discover the best model for predicting the churn rates of the company in question. The further limits and challenges of this research will also be discussed. 

## Literature Review

Customer churn prediction has always been an integral aspect of any successful business model, however, in recent years academics and business owners have been working together to place more emphasis on the study of this topic. Currently, a considerable amount of research has been focused on the telecommunications industry as this industry experiences very high amounts of customer churn. However, all industries experience churn and even models created for distinct industries can be useful in understanding and optimizing certain customer churn problems as they occur in other industries. Predicting customer churn requires a combination of industry-specific knowledge for selecting the best variables and attributes as well as the correct machine learning models that will work for that specific set of data and that specific industry.

The starting point for any analysis of customer churn modelling is on the subject of how to best select attributes or features of a given organization to be used in the model for predicting churn. Thus, feature selection plays a very important role in the early stages of data preparation. Ahmad et al. (2019) conducted a study to try to predict customer churn in a telecommunications company. This study used tree-based machine learning algorithms as they would not be affected by imbalanced data and used AUC (area under the curve) as an evaluation metric. Of the algorithms used, XGBoost appeared to be the most effective model, achieving the highest results across the metric. The next best models were the random forest model followed by the decision trees model. To ensure these models would be as successful as possible, Ahmad et al. (2019) then attempted to select the most effective variables. This was done using various industry standard selection techniques such as feature engineering, effective feature transformation, selection approach, and specific social network analysis features. It was found that the social network analysis technique enhanced the performance of their model by over 10%. Similarly, a study by Lemos et al. (2022) was conducted to explore customer churn rates in the banking industry. They analyzed various machine learning techniques including support vector machines (SVM), k-nearest neighbors (KNN), logistic regression, decision trees, random forests, and an ensemble method. Lemos et al. (2022) found that the most effective models in the study of this industry were a random forest model and an ensemble method model. This was calculated using the area under the curve method which produced a score of 90%. A feature selection analysis was applied to determine what attributes had the strongest potential for predicting customer churn in the banking industry. This feature selection method was determined by taking the average classification of decision trees, logistic regression, and elastic net algorithms and ranking them to find which variables were most important. They found that the variable pertaining to a customer's credit score had the highest statistical power. Coussement and den Poel (2008) performed an analysis on the media industry, specifically a newspaper subscription service. This study particularly focused on the use of SVM in customer churn prediction as it was expected to perform well with these features of this industry. Random forest and logistic regression techniques were also calculated in order to provide a baseline. The study found that SVM techniques have better predictive capabilities than logistic regression techniques but are outperformed by random forest techniques. They found that SVM performs best when paired with parameter selection techniques. The attributes found to be most important in the model are attributes describing the subscription and client/customer relationship features. Subsequently, Coussement and Bock (2013) performed another similar analysis of predicting customer churn in the online gambling industry. The study compares ensemble models and singular machine learning techniques and finds that ensemble models are more effective. Miguéis et al. (2012) perform an analysis in the grocery-retail industry where they apply various logistic regression techniques against the standard random forest modelling. This study uses specific variable selection techniques which have a time component within and thus lend to a chronological order model and a reverse chronological order model for the variables. Both of these logistic regression models resulted in an AUC of 86% while the random forest model resulted in an AUC of 85%, the difference was found to be significant. 

The general consensus in the literature pertaining to customer churn rate prediction points to few machine learning algorithms which tend to be most effective and most used in this industry. These models are support vector machines, naïve bayes, random forest, logistic regression, decision trees, and ensemble methods. Owczarczuk (2010) suggests that linear models are the most stable in predicting customer churn. This was discovered in their study to analyze the churn rates for the prepaid sector of a telecommunication company. Owczarczuk (2010) finds that decision tree models age very quickly and therefore their performance weakens gradually due to time. This is displayed in the fact that logistic regression models perform similarly to decision trees when looking in the short term but perform significantly better in the medium term. Similarly, Schaeffer and Sanchez (2020) looked to explore the general phenomenon of sectors in which prepaid services occur, as this type of relationship with companies often leads to high churn rates. It was found that for this type of customer churning, random forests produced the best prediction model, providing a specificity rate of up to 92%. SVM modelling was also found to produce adequate results while an ensemble method was not successful. Nie et al. (2011) performed a study in the banking sector where they analyzed credit card information from a Chinese bank. The study used logistic regression and decision trees to determine the best model. They found that the two standard types of errors were not effective in explaining the performance of these two models, instead, they developed their own measure of effectiveness which considers the economic cost in the evaluation. In considering this, they found that logistic regression performed better than decision tree models. Adaji and Vassileva’s (2015) research explores customer churn prediction in the social networking industry, they specifically looked at a case study of a popular question and answer website. To do so they employed the machine learning models of logistic regression, SVM, neural networks, and random forest to attempt to discover which users would leave their website. It was found that random forest had the highest prediction accuracy score of 76% and logistic regression had the lowest prediction accuracy of 58%. These models were evaluated using precision, recall, ROC, and percentage correctly classified (PCC) metrics. 

Deep learning and boosting techniques are also occasionally applied to attempt to predict customer churn. A study by Jain et al. (2020) attempted to predict the customer churn rates in the telecommunications industry using two machine learning techniques, logit boost and logistic regression. Many measures were analyzed to evaluate the performance, but it was found that these two techniques did not outperform one another, both having an accuracy of around 85.1% and 85.2% respectively. Vafeiadis et al. (2015) performed an analysis on some of the most popular industry-leading churn prediction techniques which included support vector machines, naïve bayes, logistic regression, back-propagation network, and decision trees. They found that the best performing techniques were back-propagation and decision trees which both achieved a measure of 94% accuracy and an f-measure of 77%, the SVM technique followed closely behind in success rates. This study provides insights into the potential usefulness of boosting techniques, as when boosting was applied to the appropriate techniques the accuracy scores increased between 1% and 4%. Siddika et al. (2021) conducted an analysis on the telecommunications industry to explore the effects of both machine learning and deep learning techniques. This study compared several models such as random forest, naïve bayes, logistic regression, KNN, and decision trees with the deep learning models of convolutional neural networks (CNN) and multilayer perceptron (MLP). All were evaluated using the same metrics and it was found that random forest produced the highest accuracy obtaining a score of 94.66%, outperforming all machine learning and deep learning techniques.

A large amount of literature tends to find that combining multiple machine learning models through ensemble and hybrid methods acquires the strongest results when comparing these models to singular model methods. Hu et al. (2020) combine decision tree and neural network models in an attempt to create a superior prediction formula. As they describe, the combination of these two models takes the confidence found in decision tree models with the weighting used in neural networks. The prediction accuracy for the combined decision tree neural network model is 98.87%, higher than both models conducted separately. Reddy et al. (2022) recently proposed an ensemble method combining random forest, XG boost, KNN, logistic regression, and stacking methods to attempt to predict customer churn rates in online shopping in India. This model was evaluated using accuracy, precision, recall, and F1score metrics. This model produced an accuracy score of 90.65%. Beeharry and Fokone (2022) performed a study on an e-commerce company in which the main purpose of their research was to determine if any single machine learning algorithm would produce stronger results than multiple algorithms combined, which it was not. The multiple algorithms combined method produced an accuracy score of 71%.

The latest research shows that these combined and non-combined methods alike can be most improved by the use of k-means clustering as a first step. Machado and Karray (2022) present findings discussing the efficiency of hybrid machine learning models for customer churn analysis using k-means clustering. The hybrid models occur by first using an unsupervised machine learning algorithm followed by a supervised machine learning algorithm. This study began by implementing k-means clustering algorithms onto the variables and then used random forest, decision trees, SVM, artificial neural networks, adaboost and gradient boosting methods to attempt to find the model which could best predict the customers who were the largest churn risk. Machado and Karray (2022) found that the hybrid model outperformed all singular models. The hybrid model of k-means and decision trees, k-means and random forest, and k-means and SVM performed best all having explained variance (EV) over 80%, indicating a strong prediction power. Xiahou and Harada (2022) considered this same theory applied to customer churn modelling in the e-commerce sector. They performed a study using k-means clustering to group the customers into core groups after which support vector machine algorithms were applied. This combination of k-means clustering and SVM techniques was found to be successful, as the k-means clustering was significant and the SVM techniques performed stronger than logistic regression techniques. This study used AUC metrics to evaluate the models. A study conducted by Ullah et al. (2019) produced similar results when exploring the use of k-means clustering before applying random forest algorithms.

The pieces of literature summarized above produce interesting and varying results. The model which appears most reliable is the random forest algorithm as it produces a very successful model in the majority of studies it is applied in. It performed well against other machine learning models, ensemble methods, and even deep learning techniques (Adaji and Vassileva, 2015; Ahmad et al., 2019; Coussement and den Poel, 2008; Lemos et al., 2022; Schaeffer and Sanchez, 2020; Siddika et al., 2021). Logistic regression modelling appears to be quite polarizing, either being very effective when applied to some sectors or very ineffective when applied to others (Adaji and Vassileva, 2015; Jain et al., 2020; Miguéis et al., 2020; Nie et al., 2011;  Owczarczuk, 2010). The other machine learning algorithms produce differing results and could be dependent on variable selection techniques and industry. Future research on the differences between models employed in these various industries would be of interest. 

## Methodology

#### Data Acquisition

The data for this research was collected from a data repository on the website Kaggle. This data was uploaded and updated by user Pawan Trivedi in April 2022 under the name of Customer Churn (see References for link to dataset). This data originates from a machine learning competition on the website HackerEarth under the title HackerEarth’s Machine Learning challenge: How NOT to lose a customer in 10 days. It explains that the data set contains information regarding the demographics, membership details, duration and frequency of visits to the website, grievances, and feedback of an up-and-coming start-up (HackerEarth, 2021). This data set was taken from Kaggle instead of directly from HackerEarth as the data set from Kaggle provides a simplified variable of churn risk score in which the churn risk score is binary, rather than on a scale of 1 to 5. Overall, there were 56,911 data points provided. 

This set of data contains 24 distinct variables. Of these 24 variables, 21 will be included in the calculation of customer churn rates. The target variable is churn risk score, which is operationalized by a 0 or a 1. A score of 0 implies that a customer did not churn and a score of 1 implies that a customer did churn. The total proportion of this variable shows that 54% of people churned and 46% of people did not churn.  Of the original 24 variables within the data set, three variables were discarded due to lack of relevancy, these variables were, customer security number, referral ID, and customer number. Further, summarized below in Table 1 are the 20 other variables which were selected to be used in the calculations. 









<sub> Table 1: Summary table of the 21 variables and their descriptions extracted from the Customer Churn data set </sub>

| Variable |	Description |
| -------  | -------- |
| Age |	10 to 64 |
| Gender |	Gender to which customer identifies  |
| Region category |	Region to which customer belongs (city, town, or village) |
| Membership category |	Type of membership customer is subscribed to |
| Joining date |	Date customer joined website |
| Joined through referral |	If the customer joined by referral |
| Preferred offer types |	Customer’s preferred way to receive offers |
| Medium of operation |	Type of device used to operate website |
| Internet connection |	Type of internet connection used to operate website |
| Last visit time |	The last time the customer visited the website |
| Days since last login |	Number of days since last login to website |
| Average time spent |	Average time spent on website |
| Average transaction value |	Average amount spent on a transaction |
| Average frequency login |	Average number of days a customer has logged into website |
| Points in wallet |	Number of points the customer has received from each transaction |
| Special discount used |	If a customer has used a special discount |
| Offer application preference |	If a customer prefers to use offers |
| Past complaint |	If a customer has had any complaints |
| Complaint status |	How a complaint has been dealt with |
| Feedback type |	Reason for complaint |

In preparation for the calculations, this data was cleaned. During cleaning it was discovered that a small amount of data was missing for some variables in this data set. The missing data was located in the variables of region category, joined through referral, medium of operation, points in wallet, and preferred offer types. The variable which contained the largest amount of missing data was joined through referral, which was missing 5438 data points which is 15% of the total data. Missing data points were defined as any data points which contained empty cells or were titled “NA”.  Ultimately, this missing data was dealt with by having any rows containing this criterion removed from the data set.  No standardization was applied to the data at this time as the majority of variables were categorical and standardization would only be applied if required by each unique machine learning technique.

The calculations for this analysis were conducted using R programming language. Various built-in R packages were required and used to optimize results. Excel spreadsheets were required for reviewing the data set, performing calculations and to allow it to be converted into a CSV file.

#### Data Preparation

To prepare the data to be input into machine learning algorithms a further analysis was performed on the variables to determine their usability. The variables last visit time, average frequency login days, and average time spent (on website) were all removed from calculations due to their lack of context and practicality. The variable offer application preference was also removed due to its similarity to other variables. This resulted in 16 variables being used for the analysis.

All missing or incorrectly spelled data was filtered and removed from the dataset, as previously discussed. However, certain variables required further filtering. The variable days since last login contained a high number of observations numbered “-999”, this was potentially a database error accounting for a lack of information. Due to this having no meaning, all rows containing this number were removed. Additionally, approximately 25 observations contained “unknown” as a gender. Due to the small number of instances of “unknown” in this variable, these 25 rows were removed from the data set as well. After these adjustments, the data set contained 19,550 data points of 16 variables. Finally, due to computational power restraints, the data set was further paired down to 12,000 observances. This was completed by using the first 12,000 observances of the data set, which shared the same proportions (54% of people churned and 46% of people did not churn) of the target variable as did the larger data set. These 12,000 observances were split into training and testing groups for each machine learning model. For all models, the observances were split into 80% training and 20% testing. Numerically, that is 9600 observances and 2400 observances respectively. 

The variable joining date was manipulated to allow for easier computation. Originally, this variable contained 1096 unique dates between the years 2015 to 2017. This was reduced to 12 levels, grouping each date quarterly by year, for example, all dates between April 1st and June 30th 2016 were reduced to the quarter, April to June 2016. 

In order to run a support vector machines algorithm, encoding of the categorical variables was necessary due to the requirement that all variables be in a numeric class in order to be able to have calculations performed on them when using this algorithm. To ensure that the categorical variables did not lose their meaning when undergoing this conversion, it was necessary for them to be put in a meaningful order. 3 of these variables were simple “yes” or “no” response variables and thus no order was required, “no” was converted to 1 and “yes” was converted to 2. The variables which underwent this manipulation were joined through referral, past complaint, and used special discount. The gender category followed a similar conversion in which “F” was converted to 1 and “M” was converted to 2. The variable region category was ordered from smallest population size to largest, meaning “village” was converted to 1, “town” to 2, and “city” to 3. Similarly, the variable preferred offer types was converted numerically to reflect smallest monetary value to largest monetary value, “without offers” to 1, “Gift Vouchers/Coupons” to 2, and “Credit/Debit Card Offers” to 3. Medium of operation was converted based on useability, with “smartphone” being converted to 1, “desktop” being converted to 2, and “both” being converted to 3. The variable membership category was ordered from the cheapest membership to the most expensive membership. That being, “no membership” as 1, “basic membership” as 2, “premium membership” as 3, “silver membership” as 4, “gold membership” as 5, and “platinum membership” as 6. The feedback variable was measured by seriousness of complaint starting with “no reason specified” as 1 and then progressing from negative feedback to positive feedback. The numeric values of the rest of the variables are as follows with “poor product quality” as 2, “poor customer service” as 3, “poor website” as 4, “too many ads” as 5, “user friendly website” as 6, “products always in stock” as 7, “quality customer care” as 8, and “reasonable price” as 9. The final category affected by encoding was joining date and this category was encoded from oldest quarter to newest quarter, with “Jan to Mar 2015” as 1 and the final quarter “Oct to Dec 2017” as 12. These variables along with the remaining 4 variables already in numeric format were then combined and normalized using min-max scaling. The target variable, churn risk score was not included in this normalization. The variables were then prepared to conduct SVM analysis.

Finally, a correlation analysis was run on the 15 variables. The variables feedback, average transaction value, points in wallet, and membership category all had mild negative correlations with the target variable of customer churn. Most variables had no significant correlation between each other. Feedback and average transaction value had a positive correlation of 0.22 while points in wallet and membership category had a positive correlation of 0.18. The remainder of the variables had weaker correlations within themselves and with the target variable. All variables were moved forward for calculations as the majority of machine learning models being used in this study are non-linear and therefore a weak correlation does not imply a poor effect. Logistic regression analysis which is linear in nature will still be run for comparison. More specific feature selection was conducted in some machine learning algorithms to create further improved models.

#### Model Selection

As justified by the literature, the following supervised machine learning techniques were selected to model customer churn. These techniques were applied due to their ability to work efficiently with discrete and binary data. 

Logistic regression was used due to its simplicity and interpretability. Calculations using this algorithm can be done quickly and effectively. The decision tree algorithm was also applied to the data set. This algorithm is widely used in understanding categorical problems (Bonaccorso, 2018). It is completed by first building a decision tree based on the data provided, followed by testing the tree which was built by classifying the remaining data (Bonaccorso, 2018). Building off of decision trees is the random forest algorithm, which was also used in this study and is useful as it does not consider all variable predictors, as to not allow the strongest predictor to interfere with the investigation (Bonaccorso, 2018). This allows many splits in the tree to occur, which could help generate a stronger result. Similar to random forest, the machine learning algorithm adaboost was used in this study. Adaboost is useful in the opposite way to random forest as it does not let the presence of weak classifiers interfere in the overall model, it instead combines them to create a stronger predictor (Bonaccorso, 2018). Finally, the algorithm of support vector machines was applied. It works by assigning the data into two different classes. It does this by creating a large gap as wide as possible so that assignment is easy (Bonaccorso, 2018). The literature has shown that it can be very effective when applied to binary churn data and thus is expected to produce an interesting result.

A k-means clustering analysis was performed on the data based on the literature indicating that this leads to a boosted model (Ullah et al., 2019; Machado and Karray, 2022; Xiahou and Harada, 2022). This algorithm requires all variables to be numeric in class and was therefore able to be completed due to the encoding of the categorical variables for their use in SVM algorithms. Despite this, k-means clustering is an unsupervised machine learning technique and therefore was not effective on this particular data set. It was thus abandoned and no further analysis was able to be completed with it, thus it is not included in the results section.

#### Metric Selection

The machine learning models will be evaluated using numerous metrics to ensure robustness and accuracy. First, confusion matrices will be applied to each model. The confusion matrix generates the metrics giving true positives, false positives, true negatives, and false negatives. This allows for calculations of precision, accuracy, recall, and F1 score. This study is most interested in the metrics of accuracy, detecting a high number of correct predictions, and F1 score, which takes into consideration both precision and recall (Novakovic et al., 2017). F1 score maximizes both of these metrics to create a more thorough picture of the prediction (Novakovic et al., 2017). Therefore, further to this, this study will also focus on the metric of sensitivity rather than specificity. This is because in this case, it is more important to be able to correctly identify when the condition is present (customer did churn). This is preferable to the alternative that the model focuses more on being able to detect when a customer won’t churn, as a company is better off targeting customers who are going to stay already rather than not targeting some customers who are planning to leave. These metrics are in line with the literature previously discussed, of which the majority use these metrics to ensure the best model. The formulas for each of these metrics are as seen below (Ullah et al., 2019).


_<p style="text-align: center;"> Accuracy =( True Positives+True Negatives)/(Total Observations) </p>_
_<p align="center"> Precision = (True Positives)/(True Positives+False Positives) </p>_
_<p align="center"> Recall/Sensitivity = (True Positives)/(True Positives+False Negatives) </p>_
_<p align="center"> F1 Score = 2×((Precision×Recall)/(Precision+Recall)) </p>_


Other metrics will also be considered .Two of these metrics are Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC). The ROC curve serves the purpose of summarizing all of the confusion matrices which were produced and aims to reach the false positive rate as fast as possible (Zheng, 2015). The AUC curve builds upon the ROC curve as it allows for ROC’s to be compared to one another, the larger the AUC curve the stronger the model (Zheng, 2015). Additionally, the metric akaike information criterion (AIC) will be applied to logistic regression models for evaluation. This metric does not provide useful information for any other model, as it was only used for logistic regression, to explore which model is the best fit (Dean, 2014). 

Following metric calculations, the various models will be compared against one another to find the strongest model. The various different metrics are used to allow for a wide understanding of each model. The comparison between these models will ultimately result in one strongest model which will be applied to understand this company’s customer churn.

## Results

#### Logistic Regression

The first of the machine learning algorithms to be executed were the logistic regression models, of which three models were produced, see Box 1 below for the basic R command for these models. Logistic regression model 1 produced a high accuracy score of 83.67% and a high AUC score of 0.9599. The sensitivity score and f1 score of 0.6765 and 0.7614, respectively, performed moderately. The AIC score for this model was 4744.8. These metrics were calculated using the confusion matrix as seen in Box 2.

Following the completion of this model, an odds ratio analysis was run to determine what variable instances had the strongest association with predicting churn risk score. Logistic regression model 1 established that multiple levels in the variable feedback had a low association with the outcome, including friendly website, price, customer care, and always in stock. The membership categories of platinum membership and premium membership were also found to have the same effect of low association, meanwhile no membership had a high association with the outcome. Two joining dates also saw a high association with the outcome, those being people who joined in Oct to Dec 2016 and those who joined in Apr to Jun 2017. These variables possessing strong associations (in either direction) were used in an attempt to predict a second improved model. The variables used in logistic regression model 2 were joining date, membership category, and feedback. This new model resulted in a slightly higher AIC score of 4858, a slightly lower AUC score of 0.9480, and a lower sensitivity score of 0.6479, when compared to model 1. While the accuracy and sensitivity scores in model 2 saw very small similar scores as model 1. 

To explore if any of these variables were the sole connection, each of these variables were executed in their own models. The models using the variables joining date and feedback resulted in very high AIC scores and low scores in the other metrics. They were thus not included in Table 2 and were not given a further analysis as this would not be an improvement. The final model, logistic regression model 3 was calculated using the variable membership category.  This model did obtain a much lower AIC score than the other sole variable models, however, it’s AIC score did not improve upon logistic regression model 1 or logistic regression model 2. In addition, logistic regression model 3 performed less well in all other metrics, obtaining the lowest AUC score, f1 score, sensitivity score, and accuracy score of all models. A summarization of all metrics from each model can be seen below in Table 2. Additionally, see Figure 1 for a comparison of ROC curves from each model.

<sub> Box 1: Logistic regression model 1 R command </sub>
```
mylogit1 = glm(churn_risk_score ~ ., data = churn4_tr, family = "binomial")
```


<sub> Box 2: Logistic regression model 1 confusion matrix </sub>
```
> confusionMatrix(table(churn_pred1, churnrisk_test))
Confusion Matrix and Statistics

           churnrisk_test
churn_pred1    0    1
          0  757   30
          1  362 1251
```

<sub> Table 2: Logistic regression metrics </sub>

| Model  |	AIC  |	Accuracy |	Sensitivity |	F1 Score |	AUC |
| ---- | ----  | ---- | --- | ---- |--- |
| Logistic Regression Model 1 |	4744.8 |	0.8367 |	0.6765 |	0.7614 |	0.9599 |
| Logistic Regression Model 2 |	4858 |	0.8358 |	0.6479 |	0.7648 |	0.9480 |
| Logistic Regression Model 3 |	5475.8 |	0.7783 |   	0.5246 |	0.7066 |	0.9305 |

 
<sub> Figure 1: Logistic regression ROC curves </sub>
<img width="332" alt="image" src="https://user-images.githubusercontent.com/77642758/188470963-650baa52-e18f-4eea-aad1-6348332593f6.png">


#### Decision Tree

The decision tree algorithm was the next machine learning model to be applied to the customer churn prediction data. It was first necessary to explore the attribute usage metrics executed by each model. In the first two models, decision tree model 1 and decision tree model 2, it was found that membership category was the most used attribute at 100%, followed by points in wallet at 58.17% and feedback at 19.91%. All other attributes had attribute usage below 10% for the first two models. The third model had the three previously mentioned attributes all at 100% attribute usage, followed by 8 of 12 remaining variables having attribute usage between 15% and 75%, allowing for a much more diverse spread. The final model, decision tree model 4, had all attribute usage above 59%.

Decision tree model 1 included all 15 predictor variables and resulted in a tree size of 106. This created a high accuracy score of 94.08% and had both a high F1 score and a high sensitivity score where both metrics were over 0.89 (see Box 4 for confusion matrix and exact summary statistics). Finally, decision tree model 1 resulted in a high AUC score of 0.9782, see Figure 1 for the accompanying ROC curves for all decision tree models.

Continuing, in an attempt to improve upon the previous model, a pruned tree was applied in order to reduce the possibility of over-fitting. This model, decision tree model 2, used a minimum of 9 cases and produced a tree size of 33. It performed less well compared to the first model in which it produced an accuracy score of 0.9388, sensitivity score of 0.9018, an F1 score 0.8949, and an AUC score 0.9786. 

 A third model was trialed with the use of boosting, in which it used 10 boosting iterations and resulted in a tree size of 31.6. This model performed similarly to decision tree model 2 in which it resulted in an accuracy score of 0.9396, a sensitivity score of 0.9198, and an F1 score of 0.8949. The area under the curve analysis produced a high score of 0.9782, the same score as produced by decision tree model 1.

A final decision tree trial was executed in which decision tree model 4 used a boosted model of 100 iterations and produced a tree size of 46, the R command executed for this model can be see below in Box 3. This resulted in slightly better scores across all measures compared to the first decision tree model, see Box 4 for the confusion matrix and various summary statistics. Decision tree model 4 resulted in the highest accuracy score of 0.9421, the highest sensitivity score of 0.9225, and the highest F1 score of 0.8990. However, it produced a slightly lower AUC score of 0.9774 when compared to the first and third decision tree models which produced an AUC score of 0.9782.

<sub> Box 3: Decision tree model 4 R command </sub>
```
churn_dt_boost100 <- C5.0(churn_dt_train[-16], churn_dt_train$churn_risk_score, control = C5.0Control(minCases = 9), trials = 100)
```

<sub> Table 3: Decision tree metrics </sub>

| Model |	Accuracy |	Sensitivity |	F1 Score |	AUC |
| ---- | ----- | ----- | ---- | ----- |
|Decision Tree Model 1 |	0.9408 |	0.9207 |   	0.8970 |	0.9782 |
|Decision Tree Model 2 |	0.9388 |	0.9018 |	0.8949 |	0.9786 |
|Decision Tree Model 3 |	0.9396 |	0.9198 |	0.8949 |	0.9782 |
|Decision Tree Model 4 |	0.9421 |	0.9225 |	0.8990 |	0.9774 |

 
<sub> Figure 2: Decision tree ROC curves </sub>

<img width="468" alt="image" src="https://user-images.githubusercontent.com/77642758/188470746-0812eadc-0182-4e2a-aefe-26e6360326a9.png">









<sub> Box 4: Decision tree model 4 confusion matrix </sub>
```
> confusionMatrix(table(churn_dt_boost_pred100, churn_dt_test$churn_risk_score))
Confusion Matrix and Statistics
                      
churn_dt_boost_pred100    0    1
                     0 1024   53
                     1   86 1237
                                               
               Accuracy : 0.9421               
                 95% CI : (0.932, 0.9511)      
    No Information Rate : 0.5375               
    P-Value [Acc > NIR] : < 0.00000000000000022
                                               
                  Kappa : 0.8833               
                                               
 Mcnemar's Test P-Value : 0.006644             
                                               
            Sensitivity : 0.9225               
            Specificity : 0.9589               
         Pos Pred Value : 0.9508               
         Neg Pred Value : 0.9350               
             Prevalence : 0.4625               
         Detection Rate : 0.4267               
   Detection Prevalence : 0.4487               
      Balanced Accuracy : 0.9407               
                                               
       'Positive' Class : 0   
```

#### AdaBoost

The next machine learning algorithm to be executed was an adaboost algorithm which was run using the package “adabag” in RStudio. It utilized the default parameter of 100 iterations.  Based on the confusion matrix (see Box 6), this algorithm resulted in an accuracy score of 0.9407, a sensitivity score of 0.9619, and an F1 score of 0.8967. An AUC score was not able to be calculated for this algorithm. See Box 5 for the R command used to run this model.

<sub> Box 5: AdaBoost model 1 R command </sub>
```
bst <- boosting.cv(churn_risk_score ~ ., data = churn_rf_rand, mfinal = 50)
```

<sub> Table 4: AdaBoost metrics </sub>

| Model  |	Accuracy |	Sensitivity |	F1 Score |	AUC |	Error |
| ---- | ---- | ---- | ---- | --- | ---- |
| AdaBoost Model 1 |	0.9407 |	0.9619 |	0.8967 |	N/A |	0.0593 |

<sub> Box 6: AdaBoost model 1 confusion matrix and error </sub>
```
> bst$confusion
               Observed Class
Predicted Class    0    1
              0 5109  245
              1  467 6179
> bst$error
[1] 0.05933333
```

#### Random Forests

The fourth machine learning algorithm to be applied in an attempt to predict customer churn was random forest. One of the command lines written to execute this algorithm can be seen below in Box 7. The first model, random forest model 1, resulted in a high accuracy score of 0.9396, a high sensitivity score of 0.9180, a high F1 score of 0.8950, and finally a high AUC score of 0.9763.

 Random forest model 2 saw an improvement on these scores in all metrics, where the accuracy score was 0.9421, the sensitivity score was 0.9252, the F1 score was 0.8988, and the AUC score was 0.997.  The confusion matrix and summary output for this model can be seen in Box 8.

A third and final model was run to weigh costs, in an effort to further improve the algorithm. However, this model resulted in lower scores for almost every metric when compared to all other random forest models (see Table 5). The only metric which performed effectively was area under the curve which produced a score of 0.9766. The comparison between the ROC curves of all models can be seen below in Figure 3.

A variable importance plot was calculated for all random forest models and all models determined that points in wallet was the most deterministic variable followed by the variable membership category (see Figure 4).

<sub> Box 7: Random forest model 2 R command </sub>
```
rf2 <- train(churn_risk_score ~ ., data = churn_rf_trn, method = "rf",
             metric = "Kappa", trControl = ctrl,
             tuneGrid = grid_rf)
```








<sub> Table 5: Random forest metrics </sub>

| Model  |	Accuracy |	Sensitivity |	F1 Score |	AUC |
| ------  | -----  | ------ | ----- | ----- |
| Random Forest Model 1 |	0.9396  | 	0.9180 |	0.8950 |	0.9763 |
| Random Forest Model 2 |	0.9421 |	0.9252 |	0.8988 |	0.9770 |
| Random Forest Model 3 |	0.8275  |  	0.6270 |	0.7570 |	0.9766 |

<sub> Box 8: Random forest model 2 confusion matrix </sub>
```
Confusion Matrix and Statistics

   
p2     0    1
  0 1027   56
  1   83 1234
                                              
               Accuracy : 0.9421              
                 95% CI : (0.932, 0.9511)     
    No Information Rate : 0.5375              
    P-Value [Acc > NIR] : < 0.0000000000000002
                                              
                  Kappa : 0.8833              
                                              
 Mcnemar's Test P-Value : 0.02743             
                                              
            Sensitivity : 0.9252              
            Specificity : 0.9566              
         Pos Pred Value : 0.9483              
         Neg Pred Value : 0.9370              
             Prevalence : 0.4625              
         Detection Rate : 0.4279              
   Detection Prevalence : 0.4512              
      Balanced Accuracy : 0.9409              
                                              
       'Positive' Class : 0                   
```
 
<sub> Figure 3: Random forest ROC curves </sub>
 <img width="375" alt="image" src="https://user-images.githubusercontent.com/77642758/188470673-00e7fb29-a5e1-40ff-b901-5f1141ae6c8b.png">

<sub> Figure 4: Random forest model 1 variable importance plot </sub>
<img width="362" alt="image" src="https://user-images.githubusercontent.com/77642758/188470654-f0d7f13b-61e2-41f2-a816-0c929bc48404.png">


#### Support Vector Machines

Support Vector Machines (SVM) was the final machine learning algorithm to be applied to the data set for analysis. An example of the R command used to execute this code can be seen below in Box 9, which displays the code used for SVM model 2.

SVM model 1 was run using the vanilla linear kernel function, which resulted in adequate scores in all metrics in which accuracy was 79.7%, sensitivity was 0.7605, the F1 score was 0.6669, and AUC was 0.8952 (see Box 10 for the output confusion matrix). 

A second model was run using a different kernel, the gaussian radial basis kernel. This produced the strongest SVM model with an accuracy score of 81.9%, a sensitivity score of 0.7605, an F1 score of 0.6952, and an AUC score of 0.9127 (see Box 11 for confusion matrix and summary output). 

A final third SVM model was conducted using the linear kernel method. This model had a cost value grid added to it in an attempt to further improve the model. SVM model 3 resulted in nearly identical results to SVM model 1, as seen in Table 6. The main difference between these two models, is the largely decreased area under the curve score of 0.7987 for SVM model 3 compared to 0.8952 for SVM model 1. Figure 5 below which plots the various SVM models ROC curves further displays this difference in AUC scores.

<sub> Box 9: SVM model 2 R command </sub>
```
svm1 <- ksvm(churn_risk_score ~ ., data = churn4.tr, kernel = "rbfdot", type = "C-svc")
```

<sub> Table 6: SVM metrics </sub>

| Model |	Accuracy |	Sensitivity |	F1 Score |	AUC |
| ----- | -----  | ---- | ----- | ---- |
| SVM Model 1 |	79.7 |	0.7605 |	0.6669 |	0.8952 |
| SVM Model 2 |	81.9 |	0.7674 |	0.6952 |	0.9127 |
| SVM Model 3 |	79.7 |	0.7605 |	0.6669 |	0.7987 |

 
<sub> Figure 5: Support vector machines ROC curves </sub>

<img width="369" alt="image" src="https://user-images.githubusercontent.com/77642758/188470548-acedfe6d-e980-41b3-9cc7-34c4ab4e061e.png">




<sub> Box 10: SVM model 1 output </sub>

```
> table(churn4.pred1, churn4.te$churn_risk_score)
            
churn4.pred1   0   1
           0 976 300
           1 134 990
 ```

<sub> Box 11: SVM model 2 output </sub>
```
> svm1
Support Vector Machine object of class "ksvm" 

SV type: C-svc  (classification) 
 parameter : cost C = 1 

Gaussian Radial Basis kernel function. 
 Hyperparameter : sigma =  0.0395474871206944 

Number of Support Vectors : 4083 

Objective Function Value : -3660.355 
Training error : 0.164479 
> table(churn4.pred1, churn4.te$churn_risk_score)
            
churn4.pred1   0   1
           0 976 300
	     134 990
```

## Discussion

This study attempted to predict customer churn in an effort to mitigate risk for a small start-up company. To do so various machine learning models were prepared to analyze the variables involved in churn risk and create models that were best able to make this prediction. Some of the models which proved to be the most successful were unexpected as the literature indicated otherwise. To gain a full understanding of this phenomena, it is first necessary to compare models within their own algorithms, that is comparing logistic regression models with logistic regression models and random forest models with random forest models. This is important as some metrics included in this study are only applicable when comparing models within an algorithm, but not between them, an example of such a metric is AIC, which is only used in logistic regression calculations. After determining the strongest model from each algorithm, the models will be compared against one another, to ultimately attempt to find the machine learning algorithm and specific model which is most effective at predicting customer churn risk at this start-up company.

Logistic regression models are quite basic but effective in predicting categorical variables like customer churn. A key metric used in the comparison of logistic regression models is the use of the Akaike Information Criterion (AIC) metric. This is a metric that cannot be used when comparing logistic regression models with other machine learning algorithms but can be a useful metric to use when just looking at logistic regression (Claeskens and Hjort, 2008). It is useful as logistic regression does not produce certain statistics that would normally be used when looking at goodness of fit (Dean, 2014). Thus, it produces a score to make up for this and is simple in that a low score implies a better fit model (Dean, 2014). Based on this criteria, logistic regression model 1 would be the strongest model, however, other metrics must be analyzed as well. Accuracy is an important metric to observe as it is informing us how many times the logistic regression model was right. Both the first and second models have an accuracy measure of 83%, while the third model is lower. Already, in two metrics the third model is underperforming and thus is likely not the strongest model. This is likely due to the fact that this model is computed considering only 1 variable as a predictor, limiting its scope and variability. The remaining two models obtained similar F1 scores and thus this metric is not likely to give further information into the strength between these models. However, the final metric of sensitivity does show a difference. As discussed previously, we are most interested in this metric as it is providing information on the proportion of positive cases, the cases in which a person did churn. This is important because it means there are less amounts of cases which were false negatives, implying that there are fewer amounts of customers churning (leaving) from the company that go undetected. Thus, in this metric, linear regression model 1 performs the strongest and provides a sensitivity score of 67.65%. This score is adequate but not excellent, however, considering it is the highest score out of all logistic regression models it will be accepted. Ultimately, when comparing all metrics in the logistic regression models, it is clear that logistic regression model 1 is the strongest model.

The decision tree models all produced very promising and similar results. All scores in all 4 models reached levels above 85%. Model 1 and 4 of decision trees achieved the highest accuracy scores, over 94% of the time both of these models correctly classified if a customer churned. These two models also achieved similar scores in the measure of sensitivity in which both achieved scores of 92%, however, the other models also achieved very similar high results. The F1 score showed nearly identical scores across all four models. This is a generally used metric that is supposed to obtain a larger view of the model. As is in the logistic regression model, the fact that all models produced nearly identical results causes it to lose effectiveness when performing a comparison. The final metric, AUC, is thus extremely important as it will hopefully be able to determine which model is best. Decision tree model 1 and decision tree model 3 both resulted in identical AUC scores of 0.9782. While decision tree model 4 resulted in an AUC score of 0.9774. Surprisingly, decision tree model 2 which performed the worst in every other metric, performed best in the area under the curve metric, having the highest AUC score of 0.9786. Ultimately, these are all very high scores and reflect a very high accuracy in every model. Although, it is important to note that all of these scores are only around 0.0010 points different from each other, making them quite similar. Therefore, it can be determined that decision tree model 4 is the strongest model due to it obtaining the highest scores in 3/4 metrics. This is a difficult comparison as decision tree model 1 is of interest due to it also obtaining very strong scores in all metrics and being the simplest model. Decision tree model 4 is a more complicated model and runs the risk of potential overfitting, as it was boosted through the use of 100 trials. However, this is likely not occurring because it is very similar to decision tree model 3 which only contained 10 boosted trials. 

A singular adaboost algorithm was applied and obtained generally high scores throughout all metrics. The sensitivity score and accuracy score of 96% and 94% respectively, are the highest scores seen so far. This algorithm was ultimately executed and used in the study because it was expected to be a strong performer based on the literature suggesting that boosting techniques tend to enhance customer churn models (Vafeiadis et al., 2015; Jain et al., 2020; Siddika et al., 2021). However, although adaboost model 1 produced respectable scores they are not outstanding. Additionally, the lack of AUC score for comparison further inhibits this model from being successful.

Random forest model 1 and random forest model 2 both saw similar effectiveness. Model 2 displayed slightly higher percentages of roughly one percent in both the accuracy metric (94.21%) and the sensitivity metric (92.52%). F1 scores and AUC scores were found to be nearly identical. Quizzically, random forest model 3 performed poorly on all metrics except AUC in which it achieved a high score, roughly 97%. This model had a weighted cost matrix applied to it which could have influenced these lower scores. The difference between model 1 and model 2 were that model 1 used the default settings for random forest while model 2 was auto tuned using an expanded grid. This slight tweak in the calculations may have created this subtle difference between these two models and thus lead to random forest model 2 being the strongest.

Finally, the support vector machine models produced surprising results. SVM model 1 and SVM model 3 output identical results in 3 out of 4 metrics. This is attributed to the fact that they resulted in identical confusion matrix results. Despite this, the models did result in different AUC scores, which can be easily visualized when looking at the ROC curves in Figure 5. SVM model 1 achieved an AUC score of 0.8952, 10% more than SVM model 3 which achieved an AUC score of 0.7987. The differences between these models can be found in the kernel used, where SVM model 1 utilized the vanilladot linear function and SVM model 3 utilized the linear kernel method. Thus, the main similarity appears to be that they are both linear functions. This could be why they produced the mostly similar results; however, it appears that linear is not the best suited kernel for this data set as SVM model 2 which uses the gaussian radial basis kernel outperforms in every metric. This is peculiar because as previously discussed in the literature review section, some researchers including Owczarczuk (2010) hypothesize that linear models offer the most stable models for predicting customer churn. This affect was clearly not seen in this study. Model 2 obtained an accuracy two percent higher than the previous models at 81.9% and an F1 score of 0.6952 compared to the 0.6669 F1 score of models 1 and 3. However, sensitivity scores amongst the models were similar. Finally, and importantly, SVM model 2 produced the strongest AUC score of all at 0.9127. Thus, we can conclude that SVM model 2 is the strongest model, likely due to its different kernel application.

Comparing the various models across different machine learning algorithms is challenging due to different metrics being used. The overall strongest model which contains purely the highest scores amongst almost all metrics is adaboost model 1, however, this machine learning method only generated one model and thus could be prone to reliability error. Additionally, there is no AUC score for comparison, making it difficult to compare. Support vector machines does not appear to be as strong of an algorithm compared to the others as it performed significantly lower in every metric. Thus, it is appropriate to remove support vector machines from the comparison. This is surprising because as observed in the literature review, many previous studies have indicated that support vector machines provides some of the most reliable and successful results in the field of customer churn. The lack of effectiveness of this algorithm could have occurred due to errors in the encoding of the categorical variables into numerical models. The majority of variables in this data set were categorical of nature and support vector machines performs more practically with numerical data. Finally, as discovered in the study by Coussement and den Poel (2008) mentioned earlier, support vector machines perform better when parameter selection techniques are applied prior to calculation, which in this case they were not. When looking at the 3 metrics which are the same across all variables (accuracy, sensitivity, and F1 score) it becomes clear that the logistic regression model is not as effective as the other models. Although it performs objectively well on the accuracy measure, compared to the other machine learning algorithms it does not appear to be as effective. Additionally, the sensitivity and F1 score in logistic regression model 1 is significantly lower than in the remaining models of decision tree model 4 and random forest model 2. This is logical, as logistic regression often operates as a basic comparison algorithm. 

The decision tree models and random forests are extremely similar, with their accuracy scores being identical. Random forest model 2 performs slightly better when looking at sensitivity and decision tree model 4 performs slightly better when looking at the F1 score. However, these performances only differ by less than 0.0030 and thus are not different enough for interpretability.  Both models also contain an AUC score of 97%, extremely high. A minor difference occurs in that the random forest model predicts more false negatives than the decision tree model, meanwhile the decision tree model predicts more false positives than the random forest model. For the purposes of customer churn, it is better to have more false negatives than false positives as it is safer to accidentally predict that a customer will churn, than predict that a customer will not churn because if a customer is not targeted early for potential churning the business may lose out. However, the differences are minor and a larger data set would need to be applied to see if this trend would continue between decision tree and random forest. Ultimately, it could be inferred that decision tree model 4 is the strongest model, despite almost every metric performing similarly. This is due the phenomena of Occam's razor, defined as, “a scientific and philosophical rule that entities should not be multiplied unnecessarily which is interpreted as requiring that the simplest of competing theories be preferred to the more complex” in Merriam-Webster's Dictionary (n.d.). Random forest algorithms are created from multiple decision tree algorithms, therefore, it is clear that when looking for the simplest answer, it is the decision tree algorithm which acts as the foundation. The fact that both models produce the same results, points to the fact that the decision tree model is ultimately the strongest, because it achieved the same result working with less information.

Ultimately, one must be cautious as this study was limited by many factors. The company did not provide a lot of excess information other than the data set, thus it is possible that some of the variables should have been treated differently or had a different meaning. Additionally, information could have been lost in the encoding stage of this study as the relationship between the ordered variables may have been misrepresented. Finally, due to computational demands, the data set had to be paired down, this could have resulted in a loss of information or a sampling bias. Future work can investigate the use of multiple methods added together as this may reduce possibility of bias and make up for the data lost during data preparation.

<sub> Table 7: Metric comparison between algorithms </sub>

| Model |	Accuracy |	Sensitivity |	F1 Score |	AUC |
| ----- | -----  | ---- | ----- | ----- |
| Logistic Regression Model 1 |	0.8367 |	0.6765 |	0.7614 |	0.9599 |
| Decision Tree Model 4 |	0.9421 |	0.9225 |	0.8990 |	0.9774 | 
| Adaboost Model 1 |	0.9407 |	0.9619 |	0.8967 |	N/A |
| Random Forest Model 2 |	0.9421 |	0.9252 |	0.8988 |	0.9770 |
| SVM Model 2 |	81.9 |	0.7674 |	0.6952 |	0.9127 |

## Conclusion 

This study ventured to perform an in-depth analysis of the customer churn rates in a small start-up company. Customer churn has become a buzzword in the business sector and everyone wants to utilize this information to its greatest potential. This wouldn’t be possible without specialized machine learning algorithms. The literature on customer churn prediction does not paint a picture of just a singular algorithm that is the key to this type of analysis. Instead, there are many algorithms and combinations of algorithms that can be executed in order to obtain the best results. Due to this, it is necessary to try a wide range of algorithms to see which works best for the data in question. This study strived to achieve this by applying 5 different machine learning algorithms, those being logistic regression, decision trees, random forest, adaboost, and support vector machines. These classification algorithms were all attempting to answer two questions. First, is this algorithm capable of predicting customer churn? Second, if it is, then is it capable of telling us what specific variables lead to churn or what specific combination of variables lead to churn? All of the algorithms were able to produce answers to these questions and the majority of them did so with very high accuracy. The random forest and decision tree models performed the strongest, with both models achieving 94% accuracy as well as high scores in all other important metrics. Ultimately, it was determined that the decision tree algorithm created the strongest model, based on many factors. Additionally, most models found that the membership category a customer subscribes to tends to be a very important variable to look out for when trying stop churn. This is reliable information for any business and the start-up whose information was used in this study will be able to use this data to target their customers who are exhibiting churning prone behaviours and hopefully stop them from leaving before it is too late. 




## References

Adaji, I., & Vassileva, J. (2015). Predicting Churn of Expert Respondents in Social Networks Using Data Mining Techniques: A Case Study of Stack Overflow. 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA), 182–189. https://doi.org/10.1109/ICMLA.2015.120

Ahmad, A. K., Jafar, A., & Aljoumaa, K. (2019). Customer churn prediction in telecom using machine learning in big data platform. Journal of Big Data, 6(1), 28. https://doi.org/10.1186/s40537-019-0191-6

Beeharry, Y., & Fokone, R. T. (2022). Hybrid approach using machine learning algorithms for customers’ churn prediction in the telecommunications industry. Concurrency and Computation: Practice and Experience, 34(4), e6627. https://doi.org/https://doi.org/10.1002/cpe.6627

Bonaccorso, G. (2018) Machine Learning Algorithms Popular Algorithms for Data Science and Machine Learning, 2nd Edition. Birmingham: Packt Publishing Ltd.

Claeskens, G., & Hjort, N. (2008) Model selection and model averaging. Cambridge: Cambridge University Press (Cambridge series on statistical and probabilistic mathematics ; 27).

Coussement, K., & Bock, K. W. D. (2013). Customer churn prediction in the online gambling industry: The beneficial effect of ensemble learning. Journal of Business Research, 66(9), 1629–1636. https://doi.org/https://doi.org/10.1016/j.jbusres.2012.12.008

Coussement, K., & den Poel, D. V. (2008). Churn prediction in subscription services: An application of support vector machines while comparing two parameter-selection techniques. Expert Systems with Applications, 34(1), 313–327. https://doi.org/https://doi.org/10.1016/j.eswa.2006.09.038

Dean, J. (2014) Big Data, Data Mining, and Machine Learning : Value Creation for Business Leaders and Practitioners. 1st ed. Somerset: John Wiley & Sons, Incorporated (Wiley and SAS Business Ser.).

HackerEarth 2021, HackerEarth’s Machine Learning challenge: How NOT to lose a customer in 10 days, HackerEarth, viewed 31 August 2022, <https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-predict-customer-churn/> 

Hu, X., Yang, Y., Chen, L., & Zhu, S. (2020). Research on a Customer Churn Combination Prediction Model Based on Decision Tree and Neural Network. 2020 IEEE 5th International Conference on Cloud Computing and Big Data Analytics (ICCCBDA), 129–132. https://doi.org/10.1109/ICCCBDA49378.2020.9095611

Jain, H., Khunteta, A., & Srivastava, S. (2020). Churn Prediction in Telecommunication using Logistic Regression and Logit Boost. Procedia Computer Science, 167, 101–112. https://doi.org/10.1016/j.procs.2020.03.187

Lemos, R., Silva, T., & Tabak, B. (2022). Propension to customer churn in a financial institution: a machine learning approach. Neural Computing and Applications, 1–18. https://doi.org/10.1007/s00521-022-07067-x

Machado, M. R., & Karray, S. (2022). Assessing credit risk of commercial customers using hybrid machine learning algorithms. Expert Systems with Applications, 200, 116889. https://doi.org/https://doi.org/10.1016/j.eswa.2022.116889

Miguéis, V. L., Van den Poel, D., Camanho, A. s, & Falcão e Cunha, J. (2012). Modeling partial customer churn: On the value of first product-category purchase sequences. Expert Systems with Applications, 39, 11250–11256. https://doi.org/10.1016/j.eswa.2012.03.073

Nie, G., Rowe, W., Zhang, L., Tian, Y., Shi, Y. (2011). Credit card churn forecasting by logistic regression and decision tree. Expert Systems with Applications, 38(12), 15273-15285. https://doi.org/10.1016/j.eswa.2011.06.028

Novaković, J. D., Veljović, A., Ilić, S. S., Papić, Ž., & Milica, T. (2017). Evaluation of Classification Models in Machine Learning, Theory and Applications of Mathematics & Computer Science, 7(1), 39. https://uav.ro/applications/se/journal/index.php/TAMCS/article/view/158.

‘Occam’s Razor’ n.d., in Merriam-Webster’s Dictionary, Merriam-Webster, Springfield, Massachusetts, viewed 31 August 2022 <https://www.merriam-webster.com/dictionary/Occam%27s%20razor>.

Owczarczuk, M. (2010). Churn models for prepaid customers in the cellular telecommunication industry using large data marts. Expert Systems with Applications, 37(6), 4710–4712. https://doi.org/https://doi.org/10.1016/j.eswa.2009.11.083

Reddy, M. G. A., Raghavaraju, S., & Lashyry, P. (2022). Ensemble Approach on the Online Shopping Churn Prediction. 2022 6th International Conference on Trends in Electronics and Informatics (ICOEI), 1–8. https://doi.org/10.1109/ICOEI53556.2022.9776921

Schaeffer, S. E., & Sanchez, S. V. R. (2020). Forecasting client retention — A machine-learning approach. Journal of Retailing and Consumer Services, 52, 101918. https://doi.org/https://doi.org/10.1016/j.jretconser.2019.101918

Siddika, A., Faruque, A., & Masum, A. K. M. (2021). Comparative Analysis of Churn Predictive Models and Factor Identification in Telecom Industry. 2021 24th International Conference on Computer and Information Technology (ICCIT), 1–6. https://doi.org/10.1109/ICCIT54785.2021.9689881


Trivedi, P 2022, Customer Churn, electronic dataset, Kaggle, viewed 31 August 2022, <https://www.kaggle.com/datasets/undersc0re/predict-the-churn-risk-rate?select=churn.csv>

Ullah, I., Raza, B., Malik, A. K., Imran, M., Islam, S. U., & Kim, S. W. (2019). A Churn Prediction Model Using Random Forest: Analysis of Machine Learning Techniques for Churn Prediction and Factor Identification in Telecom Sector. IEEE Access, 7, 60134–60149. https://doi.org/10.1109/ACCESS.2019.2914999

Vafeiadis, T., Diamantaras, K., Sarigiannidis, G., & Chatzisavvas, K. (2015). A Comparison of Machine Learning Techniques for Customer Churn Prediction. Simulation Modelling Practice and Theory, 55. https://doi.org/10.1016/j.simpat.2015.03.003

Xiahou, X., & Harada, Y. (2022). B2C E-Commerce Customer Churn Prediction Based on K-Means and SVM. Journal of Theoretical and Applied Electronic Commerce Research, 17(2), 458–475. https://doi.org/10.3390/jtaer17020024

Zheng, A. (2015) Evaluating Machine Learning Models. 1st edition. O’Reilly Media, Inc.
