# Probability Distributions

A probability distribution is a mathematical function that describes the likelihood of a set of possible occurrences taking place. By randomly drawing samples from a distribution, called *sampling*, we can expect our random selections to match the true distribution of the data as our number of samples approaches infinity (see the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers)).

You can think of a probability distribution as a [histogram](https://en.wikipedia.org/wiki/Histogram). A histogram features an ordered list of occurrences on the x-axis (called the domain) and the number of times those occurrences are present in some experiment as the y-axis (called the range). Below is an example of a measuring the pedal length feature from the famous [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). 

![Histogram](images/histogram.png)
> source: Wikipedia (user Daggerbox)

![Histogram](images/histogram-symmetric.png)
![Histogram](images/histogram-skewed-right.png)
![Histogram](images/histogram-skewed-left.png)
![Histogram](images/histogram-bimodal.png)
![Histogram](images/histogram-multimodal.png)
![Histogram](images/histogram-uniform-symettric.png)
> source: Wikipedia (user Visnit)

## Probability Mass Function

## Probability Density Function

plt.figure(figsize=(20,10))
![WISDM Raw Data Histogram](images/histogram-wisdm-raw-data.png)

- Often quantized. For instance, iris pedal length is a continuous value, however, for practical purposes we quantize it to half-cm. 
