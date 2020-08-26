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

public class Main {

    public static void main(String[] args) throws IOException, URISyntaxException, ClassNotFoundException {

        DataFrame dataFrame = readFromCsv();

        DataFrame transformedDataFrame = transformDataFrame(dataFrame);

        LinearModel model = createAndFitRegressionModel(transformedDataFrame);

        serializeModel(model);

        LinearModel deserializedModel = deserializeModel();

        double[] yPred = predict(deserializedModel, transformedDataFrame);

        validateModel(yPred, dataFrame.doubleVector("V14").array());

        System.out.println(deserializedModel);
    }

    private static DataFrame readFromCsv() throws IOException, URISyntaxException {
        DataFrame dataFrame = Read.csv("boston-house-prices.csv");
        System.out.println(dataFrame.toString(5));
        return dataFrame;
    }

    private static DataFrame transformDataFrame(DataFrame dataFrame) {
        DataFrame featureDataFrame = dataFrame.drop("V14");
        FeatureTransform transformer = WinsorScaler.fit(featureDataFrame);
        DataFrame transformedFeatureDataFrame = transformer.transform(featureDataFrame);
        DataFrame transformedDataFrame = transformedFeatureDataFrame.merge(dataFrame.doubleVector("V14"));
        System.out.println(transformedDataFrame.toString(5));
        return transformedDataFrame;
    }

    private static LinearModel createAndFitRegressionModel(DataFrame dataFrame) {
        Formula targetColumn = Formula.lhs("V14");
        LinearModel model = RidgeRegression.fit(targetColumn, dataFrame);
        System.out.println(model.formula());
        return model;
    }

    private static void serializeModel(LinearModel model) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream("model.mlm");
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        objectOutputStream.writeObject(model);
        objectOutputStream.flush();
        objectOutputStream.close();
    }

    private static LinearModel deserializeModel() throws IOException, ClassNotFoundException {
        FileInputStream fileInputStream = new FileInputStream("model.mlm");
        ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
        LinearModel model = (LinearModel) objectInputStream.readObject();
        objectInputStream.close();
        System.out.println(model.formula());
        return model;
    }

    private static double[] predict(LinearModel linearModel, DataFrame testData) {
        double[] yPred = linearModel.predict(testData);
        System.out.println(Arrays.toString(yPred));
        return yPred;
    }

    private static void validateModel(double[] yPred, double[] yTrain) {
        System.out.println("MAD  (Mean Absolute Deviation):  " + MeanAbsoluteDeviation.of(yTrain, yPred));
        System.out.println("MSE  (Mean Squared Error):       " + MSE.of(yTrain, yPred));
        System.out.println("RMSE (Root Mean Squared Error):  " + RMSE.of(yTrain, yPred));
        System.out.println("RSS  (Residual Sum of Squares):  " + RSS.of(yTrain, yPred));
    }
}
