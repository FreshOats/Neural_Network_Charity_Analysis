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

To encode the features, OneHotEncoder from scikit-learn was used, which separated the variables across many different columns of binary outputs. This was done for all columns except for the Amounts, which remained numerical until the optimization phase. During optimization, the ask amounts were binned either numerically or were treated categorically and adjusted by the OneHotEncoder.  

Since the Special Considerations column was already binary, with a Y/N value, one of these columns was removed, as the information was redundant. By encoding the categorical features, the maximum number of unique values in any column was now limited to the number of bins accepted in the Asking Amount binning. 


--- 
## Compile, Train, Evaluate the Neural Network Model

The initial model was defined with the input features based on the number of columns in the dataframe, which changed based on which features were included and how many bins were selected for the Asking Amount. This model included 2 hidden layers with 80 and 30 nodes, respectively. Using Keras from TensorFlow, a Sequential model was deployed using Rectified Linear Unit analysis for the hidden layers and sigmoid for the output layer, as the outcome is binary. Other models tested were softmax and tanh functions within the Sequential model. 

```python 
number_input_features = len(X_train_scaled[0])
hidden_layer1 = 80
hidden_layer2 = 30

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_layer1, input_dim=number_input_features, activation='relu'))

# Second hidden layer

nn.add(tf.keras.layers.Dense(units=hidden_layer2, activation='relu'))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Check the structure of the model
nn.summary()
```

SEQUENTIAL_PLOT


The models were complied using a 'binary crossentropy' loss function and 'adam' optimizer. While fitting the models, checkpoints were saved every 5 epochs. Finally the models were analyzed for accuracy and loss, comparing the trained model to the test dataset. 

```python
checkpoint_path = "checkpoints/weights.{epoch:02d}.hdf5"

# Compile the model
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

# Create a callback that saves the model's weights every 5 epochs
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True, 
    save_freq = 5)

# Train the model
fit_model = nn.fit(X_train_scaled, y_train, epochs=20, callbacks=[cp_callback])

# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

The initial model had an outcome of 72.64% accuracy. 

---
## Optimization 

The target model performance for this application was 75% accuracy. In order to achieve this, a number of modifications were made. In the end, this model did not achieve 75% accuracy, only a maximum of 73.6%.

### Changes to the Features
- In the initial model, the application type groups were cut off for all groups with less than 200 counts:
    - Additional models were tested with cutoffs of 2000, 1500, 1200, 1000, 500, 200, 50, and 10 
    - Performance of the model degraded when groups were cut off above 1000 and below 50, with 50 being optimal

- The initial model used a cutoff of 1000 for the classification value counts
    - Additional models were tested with cutoffs of 5000, 4000, 2000, 1000, 500, 200, and 100
    - Performance of the model degraded with groups were cut off above 500 with 200 as the optimal value  

- The original model had 8747 unique asking amounts, which unmodified yielded one of the highest accuracies for the nerual network model
    - This was binned into several groupings with 2, 3, 4, 5, 6, 7, 8, and 9 buckets, adjusted to mostly group similar size groups; however, there were a substantial number of $5000 requests outnumbering all of the other requests combined
    - These buckets were treated either numerically, as the index numbers 0 - 6, or they were added as categories and encoded using OneHotEncoder

- Features were systematically dropped to observe their influence on the accuracy
    - The Affiliation and Organization had the largest impact, along providing the model with 69% accuracy
    - The Application Type also had a major impact on the model
    - The Classification, Income Amount, Use Case, Asking Amount, Special Considerations, and Status had little impact on the model
    - However, while each had little impact, removing them reduced the overall accuracy 

---
### Tuning the Model

Using Keras-Tuner, an attempt to find the optimal number of nodes and hidden layers was made, and several promising models were revealed. The input parameters to test from the keras-tuner were, number of nodes [1-100], hidden layers [1-5], and activation types [relu, tanh, sigmoid, softmax, exponential]. Each test was run with only 5 epochs to get a sense of where the model was converging to. Initially runs were up to 20 epochs and showed no big difference between the 5th and 20th. 

- An optimal accuracy of 72.8% was achieved with the following parameters: 
    - Removal of USE_CASE and AMT_ASK columns
    - 3 hidden layers with 21, 66, 91 nodes 
    - tanh activation functions for input and hidden layers
    - sigmoid activation for output

In the initial model, rectified linear unit activation was used, which uses less processing power than the tanh function. While relu can lead to dead nodes, the number of nodes in this model are high, so the impact of gradient loss is minimal. The difference between the hyperbolic tangent function and the relu functions were minimal in this model. The hyperbolic tangent (tanh) function allows for negative values, whereas relu does not - rectifying all negative values to 0. The following models, as well as several others, were tested.

NEURALNETWORK_DF 


Model 6 provided the highest consistent outcome of the models tested.


---
### Comparing to Other Machine Learning Models

Since neural networks are substantially demanding from both a coding perspective as well as computational, several other machine learning models were used to compare performance with the neural network, including Random Forests, SVM, Logistic Regression, Gradient Boosted Tree, and Combination Sampling. 

- Random Forests performed the best, outperforming the neural network both in accuracy as well as in processing time
    - To compute the same data that took the neural network over 3 minutes to process, Random Forests did this in 1.8 seconds
    - The accuracy acheived by Random Forests still remained below 75%, but was higher than any of the nn models, achieving 73.2% accuracy

- SVM achieved an accuracy of 73.0%
- Logistic Regression yielded an accuracy of 72.7%
- Gradient Boosted Tree yielded an maximum accuracy of 72.3%
- Comination Sampling fared the worst, with an accuracy of only 67.8% 


--- 
## Summary

While it is often suggested that neural networks are the best way to tackle complex data sets, they do have their drawbacks and should be considered carefully in how to use them and whether they are the most appropriate model. Had the other machine learning approaches been taken first, trying to optimize the results to 75% accuracy to no avail, deep learning would be the ideal next step due to the exponentially increased amount of processing time required to perform the analyses. In this case, the neural networks run up to 100 epochs at the optimized model still failed to outperform the Random Forest analysis. 

Different activation models were tested, different numbers of nodes and hidden layers, increasing and decreasing the complexity of the neural network model. While overfitting is always an issue, even with fewer nodes, the model consistently performed with an accuracy of about 72%. There are a few ways to further approach this optimization: 

- Using AWS for additional processing power, run the keras tuner such that it can loop through different bin sizes for the categorical data as well as process the layers and nodes from 1 to 5 hidden layers and 1 to 100 nodes for 200 to 1000 epochs; due to this computer's processing power, it took over 3 hours to run 100 epochs with only 1 bin size of each
- Utilize a Leaky Relu model or different customized activation model
- Perform PCA analysis to determine the impact of each of the parameters - though PCA is for unsupervised learning, it can be beneficial in identifying what features have a greater impact on the training model 
