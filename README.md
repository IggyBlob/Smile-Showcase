# Smile Showcase
A simple program for analyzing (native) memory allocations
of the SMILE Machine Learning Library (https://haifengl.github.io/).
It executes the following operations:
<ul>
    <li>loading data from file</li>
    <li>feature/data frame transformations</li>
    <li>fitting a regression model</li>
    <li>applying a trained regression model to predict values</li>
    <li>validating the model</li>
</ul>

## Runtime environment
This showcase has been tested on Java 8 and is intended to run on Linux x64.
All other native dependencies except those for Linux x64 have been removed.

## Datasets
The showcase leverages various datasets (ranging from ~200KB to ~400MB) to 
simulate load. 
Those datasets have been taken from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). 