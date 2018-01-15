
# Welcome to **A Gentle Introduction to Machine Learning Using SciKit-Learn**

This course focuses on a specific sub-field of machine learning called **predictive modeling.**

This is the type of machine learning that the scikit-learn library in Python was desinged for. 

>**Predictive modeling** is focused on developing models that make accurate predictions at the expense of explaining why predictions are made. 

You and I don't need to be able to write a binary classification model. We need to know how to use and interpret the results of the model. 

**Where does machine learning fit into data science?**

Data science is a much broader discipline. 

> Data Scientists take the raw data, analyse it, connect the dots and tell a story often via several visualizations. They usually have a broader range of skill-set and may not have too much depth into more than one or two. They are more on the creative side. Like an Artist. An Engineer, on the other hand, is someone who looks at the data as something they have to take in and churn out an output in some appropriate form in the most efficient way possible. The implementation details and other efficiency hacks are usually on the tip of their fingers. There can be a lot of overlap between the two but it is more like A Data Scientist is a Machine Learning Engineer but not the other way round. -- Ria Chakraborty, Data Scientist





### Machine Learning Can Only Answer 5 Questions


* How much/how many?
* Which category?
* Which group?
* Is it weird?
* Which action?

<p style="text-align:right"><i>explained well by Brandon Rohrer [here]((https://channel9.msdn.com/blogs/Cloud-and-Enterprise-Premium/Data-Science-for-Rest-of-Us)</i></p>

### sklearns way of algorithms

Here is sklearn's algorithm diagram. The interactive version is [here](http://scikit-learn.org/stable/tutorial/machine_learning_map/).
![sklearn's ML map](https://az712634.vo.msecnd.net/notebooks/python_course/v1/ml_map.png)

**Why I like scikit-learn**

- Commitment to documentation and usability

- Models are chosen and implemented by a dedicated team of experts

- Covers most machine-learning tasks

- Python and Pydata

- Focus

- Scikit-learn scales to most data problems

- <p><a href="http://oreil.ly/2n7xnVJ"> Ben Lorica on Using SciKit-Learn</a></p>


**Getting Started**

- Installing Python 

- <p><a href="https://www.continuum.io/downloads"> Anaconda Installer</a></p>

- Using Jupyter Notebooks

- Importing Libraries and Modules


# **The Model Building Starts Now...**

The word **'import'** simply means... bring in these libraries so we can use them. 

When we use the **from** keyword with import it means we are only importing one section or module. In the code below we are importing **read_csv** from pandas and **set_printoptions** from numpy. 

The word **numpy** means numerical python. 


```python
from pandas import read_csv
from numpy import set_printoptions 
```

Most machine learning tools use CSV files. A CSV fie is a comma separated values file. 
It's like an excel spreadsheet. In the learning world we get to work with very clean csv files. 

> In the real world... not so much. 

The data set we are working with was exported from a SQL Server data warehouse training database. 


In the code below we are reading a csv file. Notice that we are not specifying a path. So, how does the notebook know where the csv file is? Because we've placed it in the working directory. 

You can use the os.getwd() function to see where your working directory is. 


```python
filename = 'BBC.csv'
dataframe = read_csv(filename) 
```


```python
import  os
print os.getcwd() 
```

    C:\Users\mwest


In the code below take note of the syntax used to specify a folder NOT in the default directory. 

> The **df** is just a variable for holding the csv file. In a few cells below this one we will call the variable dataframe. 


```python
df = read_csv('datafiles/BBC.csv')
```

We can use the head() function to return the top x number of rows. 

> We can put a number in the function to return any number we like. 

For example, we can run dataframe.head(25) to return the top 25 rows. 

Now we can investigate our data. Let's see how many **attributes** and how many **observations** there are in our data set. 

> Notice we are using the alias dataframe that we used when we read in our data set. 

Using **shape** really means... how many columns in rows are in our data set.

Don't forget the dataframe is just the alias we are using. 


```python
dataframe.tail(5)
```


```python
dataframe.shape
```

The describe function gives us descriptive statistics about our data. 

- **mean** = average 
- **median** = values in the middle of the rage
- **mode** = The number which appears most often in a set of numbers

What is the average age in our data set? 

What is the maximimun number of cars owned by a person in our data set? 


```python
dataframe.describe()
```

In the dataframe below we are assigning our dataframe to an array. 


```python
array = dataframe.values
```

The code below is the trickest part of the exercise. Now, we are assinging X and y to an array. 

> That looks pretty easy but keep in mind that an array starts at 0. 

If you take a look at the shape of our dataframe (shape means the number of columns and rows) you can see we have 12 rows. 

On the X array below we saying... include all items in the array from 0 to 11. 

On the y array below we are saying... just use the column in the array mapped to the **11th row**. The **BikeBuyer** column. 






```python
X = array[:,0:11] 
Y = array[:,11]
```

We are using group by to view the distribution of values in our **BikeBuyer** column. Recall that this column is our **target variable**. It's that thing we are trying to predict. 

The really nice part about the distribution of buyers and non-buyers (the number of 1s versus 0s) is that they are balanced. 

> This is actually very important when it comes to scoring our model. Classiﬁcation accuracy is the number of correct predictions made as a ratio of all predictions made. This is the most common evaluation metric for classiﬁcation problems, it is also the most misused.

**Accuracy** is only really useful when there are an even distribution of values in a data set. The good news for us is in our data set they are nearly perfectly even. 


```python
print(dataframe.groupby('BikeBuyer').size())
```

    BikeBuyer
    0    9352
    1    9132
    dtype: int64


In the next cell let's import a model called **train_test_split** and one called **SVC**. 

Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. 

> This situation is called **overfitting.** To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. 

We are also importing a Support Vector Machine (sklearn.**svm**) If you navigate to the sklearn picture above you can see that SVC is a classifier in the labled data section. Many real world models will use classificaiton.

In sklearn it's called a SVC or support vector classifier. 




Support Vector Machines (SVMs) are a powerful supervised learning algorithm used for classification or for regression. SVMs are a discriminative classifier: that is, they draw a boundary between clusters of data.

> <p><a href="https://www.youtube.com/watch?v=eUfvyUEGMD8&t=21s"> Worlds Best SVM Overview</a></p>

Support Vector Machines are perhaps one of the most popular and talked about machine learning algorithms.


```python
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
```

In the cell below we are specifying the test size. If our test size is 25% of the data then our training size is 75% of the data. 

Most modelers choose to split their data 70/30 or 80/20. The larger portion is used to train the data. 

The seed below is so we can reproduce our results. 


```python
test_size = 0.25
seed = 7
```

In the cell below we are creating the split. Take note we are using the variables above. 

- test_size=test_size
- random_state=seed




```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
```

Next we set the model we are going to use. In the code below we are simply creating a variable or place holder called model for the SVC algorithm. 


```python
model = SVC()
```

In the cell below we are fitting our model variable to our training set. The code is clean and straightforward. For now, we call the model as is without any parameters. 

> Many of the classification models tend to do well out of the box. That means you can often get solid results without much tuning. 


```python
model.fit(X_train, Y_train) 
```

Next we use result to get the results of model.


```python
result = model.score(X_test, Y_test)
```

Lastly we print the accuracy of our model. 


```python
print("Accuracy: %.3f%%") % (result*100.0)
```

In the code below let's put it all togehter. 


```python
from pandas import read_csv
from numpy import set_printoptions
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier

filename = 'BBC.csv'
dataframe = read_csv(filename) 

array = dataframe.values
X = array[:,0:11] 
Y = array[:,11]

test_size = .30
seed = 45
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = RandomForestClassifier()
model.fit(X_train, Y_train) 

result = model.score(X_test, Y_test)

print("Accuracy: %.3f%%") % (result*100.0)

```

    Accuracy: 77.533%


In cell below let's use a different algorithim for our tests. 

> We are going to run our entire model in one cell. 

We only have to take two simple steps to substitue another model. 

- Import the new model. (from sklearn.neighbors import KNeighborsClassifier)
- Map the model variable to our new algorithm. (model = KNeighborsClassifier())

I use comments below to show what was changed. 


```python
from pandas import read_csv
from numpy import set_printoptions
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier

filename = 'BBC.csv'
dataframe = read_csv(filename) 

array = dataframe.values
X = array[:,0:11] 
Y = array[:,11]

test_size = .30
seed = 45
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = SVC()
model.fit(X_train, Y_train) 

result = model.score(X_test, Y_test)

print("Accuracy: %.3f%%") % (result*100.0)
```

    Accuracy: 78.164%

