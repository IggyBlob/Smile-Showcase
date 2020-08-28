# Smile Showcase
This Java program showcases the following aspects of the [Smile Machine Learning library](https://github.com/haifengl/smile):

- loading data from file
- transforming features/data frames
- fitting a regression model
- serializing/deserializing a model to/from binary representation using vanilla Java
- serializing a model to JSON using protostuff
- applying a trained regression model to predict values
- validating the model

This example relies on the 
[Boston House Prices data set](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/).

## Model serialization
According to the [docs](https://github.com/haifengl/smile#model-serialization), most Smile models implement the Java 
`Serializable` interface. This allows them to be serialized to binary representation using vanilla Java, which is 
demonstrated in this showcase. 

Additionally, this implementation indicates how a Smile model can be serialized to JSON using 
[protostuff](https://github.com/protostuff/protostuff). Protostuff seems to be the recommended way, as it is explicitly 
mentioned in the Smile docs in the context of (JSON) model serialization.

## Native dependencies
This showcase relies on native (C++) dependencies as Smile needs BLAS/LAPACK for its linear regression algorithms. 
Please refer to the [pom.xml](pom.xml) for details.

## Java version
This showcase has been tested on Java 8 and Java 11. While Smile seems to run fine on both Java 8 and Java 11, 
protostuff is not yet ready to be used with the Java Module System as it causes illegal reflective access warnings.

Please make sure to switch the Java version in your IDE if you
want to try it yourself.
