# What is Machine Learning?

Machine learning (ML) is a technique whereby computer programs **learn** to perform a task **by example**.<span class="marginal-note" data-info="You can see exactly what we mean by that in this browser demo: [Teachable Machine](https://teachablemachine.withgoogle.com/)"></span> This is in contrast to traditional programming, where a computer executes heuristic instructions that are explicitly "handwritten" by a programmer. In many areas, ML-based approaches have far surpassed previous state-of-the-art algorithms that have taken decades of domain-specific knowledge and research to develop. Modern machine learning techniques like Deep Learning (see [Neural Networks & Deep Learning](neural-networks-and-deep-learning.html)) have already been embedded in much of the software that we use everyday, and this rate of integration is increasing faster than ever. Machine learning is currently poised to dominate the coming decade of computing, and beyond. 

## When Should I Use It?

ML is both an active area of research and a set of practices that can be used to dramatically improve many software tasks today. It is not, however, a blanket solution to all of your programming problems. ML techniques are great for solving problems when:

- You are not an expert in the domain that you are writing code about
- A solution to your problem is difficult to describe, even out loud
- Your problem requires inputs from multiple variables or rich media
- You have a **lot** of data (see [Data is Key](data-is-key.html))

For hard problems, don't write heuristic rules. Instead learn from data. So long as you have enough data, a machine learning solution will often be easier to write and perform better than a rule-based approach.

## What Skills Does Machine Learning Require?

While bleeding-edge ML research
<span class="marginal-note" data-info='Most new ML research papers appear on [arxiv.org](https://arxiv.org/list/stat.ML/pastweek) before they are even presented at conferences and included in academic journals. Today, it has become standard to release open source software on sites like GitHub along with the papers on ArXiv.'></span>
may have a bit of a barrier-to-entry, most applications of ML require only a general comfortability with programming. The language of choice for most machine learning libraries and tools is [Python](https://learnxinyminutes.com/docs/python/). This is likely a result of the language's history in scientific computing, and its reasonable to assume that Python will remain dominant in the field for the foreseeable future. That said, the popularity of machine learning has spiked dramatically since ~2015-2016 and ML libraries are popping up in just about every language and platform imaginable (including embedded devices). The code examples in this resource will be written in Python as well as JavaScript using the new [Tensorflow.js library](https://js.tensorflow.org/).

Besides standard programming skills, ML specifically borrows from the following fields:

- Linear Algebra
- Statistics
- Probability

While experience in any of these topics will certainly be helpful as you learn and employ ML, I want to stress that prior knowledge of these fields are in no way necessary. They should be thought of more as areas of [further study](https://www.youtube.com/watch?v=kjBOesZCoqc&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) if you want to improve your ML skills and understanding.

Lastly, the OS of choice for ML projects is certainty Ubuntu/Debian Linux. While many of the most popular frameworks have cross-platform options, Linux is hands-down the most supportive OS for all things machine learning. If you are not yet comfortable with basic linux/unix commands, I recommend learning. While it is not required, you will find your productivity running ML experiments and iteratively improving ML algorithms will benefit greatly from this knowledge.

Next: [General Purpose Algorithms](general-purpose-algorithms.html)