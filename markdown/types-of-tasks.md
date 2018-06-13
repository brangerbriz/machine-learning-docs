# Types of Tasks

Before we continue, I want to hammer in the difference between two of the most common ML tasks [we mentioned before](general-purpose-algorithms.html#classification); classification and regression.

![Classification vs Regression](images/classification-vs-regression.png)
> source https://ipython-books.github.io/featured-04/

## Classification

> In machine learning and statistics, classification is the problem of identifying to which of a set of categories a new observation belongs, on the basis of a training set of data containing observations whose category membership is known. - Wikipedia

Classification problems output a probability distribution called a [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function) over a set of categories (e.g. 22% probability an email is spam, 88% probability an email is legitimate). The probability distribution produced by a classification problem sums to `1.0`, e.g. `0.22 + 0.88 = 1.0` (see [Probability Distributions](probability-distributions.html)).

## Regression

Regression problems produce real-valued floats, or rather, a distribution called a [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) over real-valued continuous floats that integrate to `1.0`.

In short, classification problems output [discrete data](https://stats.stackexchange.com/questions/206/what-is-the-difference-between-discrete-data-and-continuous-data) while regression problems output continuous data.

Next: [Performance Measures](performance-measures.html)