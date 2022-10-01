# Deep Learning in Philanthropy
***Neural Network Modeling to Identify Parameters in Successful Recipients***
#### by Justin R. Papreck
---

## Overview

The client is a foundation that has made over 34,000 charitable donations over the years, and they would like to try to identify predictors of successful donations. To date, only 53.2% of the organizations that have received money from the foundation have been successful. The purpose of this application is to use deep learning to classify and predict whether future applicants are likely to be successful. After completing this analysis, the highest accuracy achieved by deep learning was 72.7% accuracy, wheras using Random Forests achieved a slightly higher 73.3% accuracy with a much lower computational impact and processing time. 

--- 
## Methods
***Preprocessing the Data***

Most of the analysis was performed using scikit-learn, TensorFlow, and Pandas. 
- The Target variable for this analysis is the "IS_SUCCESSFUL" column, which contains only 1 or 0 numerical entries
- It was not indicated whether 1 or 0 was 'yes', so the interpretation of this analysis is that 0 is unsuccessful and 1 is successful
- The Name and EIN are neither targets nor are they features and were removed from the dataframe immediately
- The features of the model included Application Type, Affiliation, Classification, Use Case, Organization, Status, Income Amount, Special Considerations, and the Asking Amount
- Most of the features are of object data types, and must be transformed for analysis

---
### Analyzing Unique Values Per Column

Prior to formatting the data, the number of unique values within each column were determined to limit a skewing of the weights toward unimportant groupings while training. 

```python 

application.nunique()

```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>APPLICATION_TYPE</th>
      <th>AFFILIATION</th>
      <th>CLASSIFICATION</th>
      <th>USE_CASE</th>
      <th>ORGANIZATION</th>
      <th>STATUS</th>
      <th>INCOME_AMT</th>
      <th>SPECIAL_CONSIDERATIONS</th>
      <th>ASK_AMT</th>
      <th>IS_SUCCESSFUL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th></th>
      <td>17</td>
      <td>6</td>
      <td>71</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>8747</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

The columns that contain more than 10 values are good candidates to bucketing, which include the Application Type, Classification, and the Ask Amount. The values were counted for the Application Types and Classifications, grouping the classes under a particular threshold into an 'Other' bin for the Neural Network analysis. This number was originally set looking at the density plots of the Application Type and Classification, but then were systematically adjusted during optimization to yield the ideal number of bins and splits for training. Below are the Application Type plot and subsequenty the Classification density plot. 

APP_TYPE_PLOT

CLASSIFICATION_PLOT


After grouping these, the Asking Amounts were grouped into bins, again adjusted during the optimization phase to determine the optimal number and spacing of the bins using the following code: 

```python
max_ask = application_df.ASK_AMT.max()

bins = [0, 5000, 10000, 50000, 100000, 500000, 1000000, max_ask]
labels = [0, 1, 2, 3, 4, 5, 6]

application_df["ASK_BIN"] = pd.cut(application_df["ASK_AMT"], bins=bins, labels=labels)
```
 
The pandas cut function cuts the selected data into groups right-inclusive up to the max value. 

---
### Encoding the Categorical Features

To encode the features, OneHotEncoder from scikit-learn was used, which separated the variables across many different columns of binary outputs. This was done for all columns except for the binned Ask Amounts, which remained numerical. In a later analysis, the binned amounts were treated categorically, however the results actually decreased in accuracy from the previous models.  

Since the Special Considerations column was already binary, with a Y/N value, one of these columns was removed, as the information was redundant. By encoding the categorical features, the maximum number of unique values in any column was now limited to the number of bins accepted in the Asking Amount binning. 