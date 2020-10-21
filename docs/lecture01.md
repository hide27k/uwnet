---
layout: default
title: Deep Learning - 01 - Machine Learning Review
---

# Chapter 1: Machine Learning Review

So what exactly is deep learning, anyway? The phrase is fairly vague and means different things to different people depending on who you talk to. First, though, it's good to have a basic understanding of typical machine learning tasks and pipelines to understand how deep learning is different.

## Machine Learning Tasks

Machine learning broadly is the task of modelling data, usually with some kind of numerical or statistical model. The first key distinction between machine learning tasks is between **supervised** and **unsupervised** learning:

- **Supervised learning** is function approximation. 
    - Input:
        - data $$X$$
        - labels $$Y$$
        - paired examples $$(x,y)$$
    - Assume:
        - there exists a function that maps from data to labels $$f: X \to Y$$
        - our paired examples $$(x,y)$$ satisfy $$f(x) = y$$
    - Learn: approximation $$h$$ such that $$h(x) \approx f(x)$$
- **Unsupervised learning** is modelling the distribution of data
    - Input:
        - data $$X$$
    - Learn:
        - **clusters**: groupings of related data points
        - a transformation to a different feature space that preserves relationships between data points
        - a generating function or probability distribution $$g$$ such that statistically $$X$$ appears to be drawn from $$g$$: $$X \sim g$$

### Supervised Learning

Supervised learning encompasses algorithms for function approximation. Given a dataset $$X$$ and a function $$f$$ that takes elements of the dataset and produces output $$y = f(x)$$, learn a function $$h$$ such that $$h(x) \approx f(x)$$.

{% include image
    src="figs/mnist_digits.png"
    alt="Examples of MNIST digits. A 10x16 grid of handwritten digits, each row is a different digit 0-9."
    attribution="By Josef Steppan - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=64810040"
    caption="Examples digits from the MNIST dataset, 28x28 pixel images of handwritten digits 0-9. MNIST is a common benchmark for computer vision tasks although it is a fairly easy to attain high accuracies."
%}

This "labelling" function $$f$$ can be obvious, like trying to predict the price of a car from attributes of the car like the make, model, year, mileage, condition, etc. In this case the true function $$f$$ is the process a car salesman goes through to put a price on a car given those attributes. We are trying to create an approximate function $$h$$ that takes the same attributes and assigns a similar price.

For some tasks it can be more opaque, like predicting today's weather from yesterday's weather. In the case of the weather, there is an underlying physical process but it is not a clear function that only takes as input the past day's weather. In reality, the weather is determined by a function (the unfolding of the laws of physics) acting on a set of data (the physical conditions of our planet).

In the case of weather prediction, the physical conditions of the planet are what's known as a **latent variable** or hidden variable. They are not fully observed or recorded but do affect the outcome. Our model can try to account for these variables or simply circumvent them. In either case we are trying to build an approximation of a function that doesn't actually exist, we simply assume it does. There's a lot of things like that in machine learning. Don't let it bother you too much!

#### Classification vs Regression

There is often a distinction drawn between **classification** and **regression** tasks in supervised learning:

- **Classification**:
    - labels are discrete classes
    - \\(Y \in \mathbb{Z}\\)
    - example algorithm: logistic regression
- **Regression**:
    - lables are real-valued
    - \\(Y \in \mathbb{R}\\)
    - example algorithm: linear regression

It's pretty confusing that both of those example algorithms have "regression" in their name, huh? There are more complicated or mixed tasks as well that involve predicting both discrete and continuous variables. We'll worry about those later.


### Unsupervised Learning



<!--
Weather Prediction Notation:
- $$X$$: yesterday's weather
- $$Y$$: today's weather
- $$f$$: hypothetical function mapping $$X \to Y$$
- $$Y$$: physical conditions of planet/universe yesterday
- $$T$$: physical conditions of planet/universe today
- $$p$$: laws of physics, $$p: Y \to T$$
- $$w$$: interpretation of physical conditions as weather, $$w: T \to Y$$
-->


I think of deep learning as having a few specific characteristics relative to typical machine learning.

### Feature Extraction

In "normal" machine learning there is an important phase known as feature extraction. The raw data must be converted to numerical or other "features" usable by a particular statistical modelling technique.

Typically deep learning is:

- Neural network based
- Uses large amounts of data
- Incorporates feature extraction as part of the model
    - Has many "layers" of processing
    - Early layers extract simple features from raw data
    - Later layers extract complex features from simple features

Testing inline math $$a + b - \frac{10}{1}$$ does it work?

$$ w = w - \eta \frac{d}{dw} L $$



