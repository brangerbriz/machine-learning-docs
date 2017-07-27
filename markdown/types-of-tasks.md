# Types of Tasks

Before we continue, I want to hammer in the difference between two of the most common ML tasks [we mentioned before](general-purpose-algorithms.html#classification).

![Classification vs Regression](images/classification-vs-regression.png)
> source https://ipython-books.github.io/featured-04/

## Classification

> In machine learning and statistics, classification is the problem of identifying to which of a set of categories a new observation belongs, on the basis of a training set of data containing observations whose category membership is known. (wikipedia)

Classification problems output a probability distribution over a set of categories (e.g. 22% probability an email is spam, 88% probability an email is legitimate). The probability distribution produced by a classification problem sums to `1.0`, `0.22 + 0.88 = 1.0`.

## Regression

Regression problems produce real-valued floats, or rather, a probability mass distribution over real-valued continuous floats that integrate to `1.0`.

Next: [Performance Measures](performance-measures.html)