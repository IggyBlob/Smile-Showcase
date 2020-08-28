import io.protostuff.JsonIOUtil;
import io.protostuff.LinkedBuffer;
import io.protostuff.Schema;
import io.protostuff.runtime.RuntimeSchema;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.feature.FeatureTransform;
import smile.feature.WinsorScaler;
import smile.io.Read;
import smile.regression.LinearModel;
import smile.regression.RidgeRegression;
import smile.validation.MSE;
import smile.validation.MeanAbsoluteDeviation;
import smile.validation.RMSE;
import smile.validation.RSS;

import java.io.*;
import java.net.URISyntaxException;
import java.util.Arrays;

/**
 * A simple program showcasing the following aspects
 * of the SMILE Machine Learning Library (https://haifengl.github.io/):
 * <ul>
 *     <li>loading data from file</li>
 *     <li>feature/data frame transformations</li>
 *     <li>fitting a regression model</li>
 *     <li>serializing/deserializing a model to/from binary representation using vanilla Java</li>
 *     <li>serializing a model to JSON using protostuff</li>
 *     <li>applying a trained regression model to predict values</li>
 *     <li>validating the model</li>
 * </ul>
 */
public class Main {

    public static void main(String[] args) throws IOException, URISyntaxException, ClassNotFoundException {

        // load CSV training data into a Smile DataFrame
        DataFrame dataFrame = readFromCsv();

        // apply feature transformations on the DataFrame
        DataFrame transformedDataFrame = transformDataFrame(dataFrame);

        // instantiate and train a linear regression model using the read training data
        LinearModel model = createAndFitRegressionModel(transformedDataFrame);

        // serialize the linear regression model to the file system
        serializeModel(model);

        // serialize the linear regression model to JSON and print it to the console
        serializeModelJson(model);

        // re-load the serialized model from the file system
        LinearModel deserializedModel = deserializeModel();

        // use the deserialized model to predict the _TRAINING_ data
        double[] yPred = predict(deserializedModel, transformedDataFrame);

        // compare the predictions to the actual target column (V14) of the _TRAINING_ data
        validateModel(yPred, dataFrame.doubleVector("V14").array());

        // print the model to inspect its parameters
        System.out.println(">>> DUMP MODEL");
        System.out.println(deserializedModel);
    }

    /**
     * Loads the Boston house prices dataset from the file system.
     * It uses the built-in CSV parser of Smile, provided via the
     * <code>Read</code> interface.
     * @return a Smile DataFrame holding the CSV data
     */
    private static DataFrame readFromCsv() throws IOException, URISyntaxException {
        DataFrame dataFrame = Read.csv("boston-house-prices.csv");
        System.out.println(">>> READ DATA FROM CSV");
        System.out.println(dataFrame.toString(5));
        return dataFrame;
    }

    /**
     * Winsorizes a DataFrame.
     * Winsorization prevents outliers by limiting extreme feature values
     * to the 5th/95th percentile of the original distribution.
     * @param dataFrame a Smile Dataframe holding Boston house price data to be winsorized
     * @return the winsorized Dataframe; the target column (V14) remains untouched
     */
    private static DataFrame transformDataFrame(DataFrame dataFrame) {
        DataFrame featureDataFrame = dataFrame.drop("V14");
        FeatureTransform transformer = WinsorScaler.fit(featureDataFrame);
        DataFrame transformedFeatureDataFrame = transformer.transform(featureDataFrame);
        DataFrame transformedDataFrame = transformedFeatureDataFrame.merge(dataFrame.doubleVector("V14"));
        System.out.println(">>> TRANSFORM DATA SET");
        System.out.println(transformedDataFrame.toString(5));
        return transformedDataFrame;
    }

    /**
     * Instantiates and trains a Smile Ridge regression model.
     * The target column y of the model is V14.
     * @param dataFrame a Smile DataFrame holding the Boston house price training data
     * @return a fitted Smile Ridge regression model
     */
    private static LinearModel createAndFitRegressionModel(DataFrame dataFrame) {
        Formula targetColumn = Formula.lhs("V14");
        LinearModel model = RidgeRegression.fit(targetColumn, dataFrame);
        System.out.println(">>> FIT MODEL");
        System.out.println(model.formula());
        System.out.println();
        return model;
    }

    /**
     * Serializes a model in binary representation to the file system.
     * The byte stream is written to a file "model.mlm" using vanilla Java.
     * @param model the Smile model to be serialized
     */
    private static void serializeModel(LinearModel model) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream("model.mlm");
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        objectOutputStream.writeObject(model);
        objectOutputStream.flush();
        objectOutputStream.close();
        System.out.println(">>> SERIALIZE MODEL TO FILE `model.mlm`");
        System.out.println();
    }

    /**
     * Serializes a model to JSON and prints it to the console.
     * To generate the JSON representation of a model, the protostuff library is used,
     * as recommended by the official Smile docs.
     * @param model the Smile model to be serialized to JSON
     */
    private static void serializeModelJson(LinearModel model) {
        Schema<LinearModel> schema = RuntimeSchema.getSchema(LinearModel.class);
        LinkedBuffer buffer = LinkedBuffer.allocate(512);
        final byte[] protostuff;
        try
        {
            protostuff = JsonIOUtil.toByteArray(model, schema, false, buffer);
        }
        finally
        {
            buffer.clear();
        }
        System.out.println(">>> SERIALIZE MODEL TO JSON");
        System.out.println(new String(protostuff));
        System.out.println();
    }

    /**
     * Loads a binary-serialized model from the file system.
     * The model is read from a file "model.mlm" using vanilla Java.
     * @return the Smile linear regression model read from the file system.
     */
    private static LinearModel deserializeModel() throws IOException, ClassNotFoundException {
        FileInputStream fileInputStream = new FileInputStream("model.mlm");
        ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
        LinearModel model = (LinearModel) objectInputStream.readObject();
        objectInputStream.close();
        System.out.println(">>> DESERIALIZE MODEL FROM FILE `model.mlm`");
        System.out.println(model.formula());
        System.out.println();
        return model;
    }

    /**
     * Predicts test data using the given linear regression model.
     * Note that no transformations are applied on the test data.
     * If the training data set has been transformed, please reapply those
     * transformations to the test data set as well.
     * @param linearModel the Smile linear regression model to be used for prediction
     * @param testData a Smile DataFrame holding the test data to be predicted
     * @return a double vector containing the predicted values
     */
    private static double[] predict(LinearModel linearModel, DataFrame testData) {
        double[] yPred = linearModel.predict(testData);
        System.out.println(">>> PREDICT DATA");
        System.out.println(Arrays.toString(yPred));
        System.out.println();
        return yPred;
    }

    /**
     * Compares the predicted values to actual values.
     * It calculates and prints various linear regression metrics to the console.
     * @param yPred a vector of predicted values
     * @param yTrain a vector of actual values
     */
    private static void validateModel(double[] yPred, double[] yTrain) {
        System.out.println(">>> VALIDATE MODEL");
        System.out.println("MAD  (Mean Absolute Deviation):  " + MeanAbsoluteDeviation.of(yTrain, yPred));
        System.out.println("MSE  (Mean Squared Error):       " + MSE.of(yTrain, yPred));
        System.out.println("RMSE (Root Mean Squared Error):  " + RMSE.of(yTrain, yPred));
        System.out.println("RSS  (Residual Sum of Squares):  " + RSS.of(yTrain, yPred));
        System.out.println();
    }
}
