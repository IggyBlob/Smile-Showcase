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

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * A simple program for analyzing (native) memory allocations
 * of the SMILE Machine Learning Library (https://haifengl.github.io/).
 * It executes the following operations:
 * <ul>
 *     <li>loading data from file</li>
 *     <li>feature/data frame transformations</li>
 *     <li>fitting a regression model</li>
 *     <li>applying a trained regression model to predict values</li>
 *     <li>validating the model</li>
 * </ul>
 */
public class Main {

    private static final int NO_OF_PROCESSORS = Runtime.getRuntime().availableProcessors();
    private static final int NO_OF_ITERATIONS = 15000 * 3 + 7500; // ~3.5 * 6h runtime using 4 cores
    private static final List<String> FILE_NAMES = Arrays.asList(
            "bike-sharing.csv",
            "election-data.csv",
            "energydata-complete.csv",
            "geographical-origin-of-music.csv",
            "online-news-popularity.csv",
            "superconductivity.csv",
            "winequality-white.csv");

    public static void main(String[] args) throws IOException {
        System.out.printf("Press any key to start the showcase with %d threads and %d tasks...",
                NO_OF_PROCESSORS,
                NO_OF_ITERATIONS * FILE_NAMES.size());
        System.in.read();
        long startTime = System.nanoTime();
        ExecutorService executorService = Executors.newFixedThreadPool(NO_OF_PROCESSORS);
        List<Callable<Object>> callables = new ArrayList<>(NO_OF_ITERATIONS * FILE_NAMES.size());
        for (int i = 0; i < NO_OF_ITERATIONS; i++) {
            for (String fileName : FILE_NAMES) {
                Callable<Object> callable = createCallable(fileName);
                callables.add(callable);
            }
        }
        try {
            System.out.printf("Waiting for %d tasks to finish...\n", callables.size());
            executorService.invokeAll(callables);
            System.out.println("Attempt to shutdown thread pool executor...");
            executorService.shutdown();
            System.out.println("Waiting for unfinished tasks to complete...");
            executorService.awaitTermination(1, TimeUnit.HOURS);
            System.out.printf("%d tasks finished\n", callables.size());
        } catch (InterruptedException ex) {
            System.out.println("Tasks interrupted");
        } finally {
            if (!executorService.isTerminated()) {
                System.out.println("Cancelling non-finished tasks forcibly...");
            }
            executorService.shutdownNow();
            System.out.println("Thread pool executor shut down");
        }
        long stopTime = System.nanoTime();
        System.out.println("Press any key to exit...");
        System.in.read();
        System.out.println(stopTime - startTime);
    }

    private static Callable<Object> createCallable(String fileName) {
        Runnable runnable = () -> {
            try {
                // load CSV training data into a Smile DataFrame
                DataFrame dataFrame = readFromCsv(fileName);

                // apply feature transformations on the DataFrame
                DataFrame transformedDataFrame = transformDataFrame(dataFrame);

                // instantiate and train a linear regression model using the read training data
                LinearModel model = createAndFitRegressionModel(transformedDataFrame);

                // use the deserialized model to predict the _TRAINING_ data
                double[] yPred = predict(model, transformedDataFrame);

                // compare the predictions to the actual target column (V14) of the _TRAINING_ data
                validateModel(yPred, dataFrame.column(dataFrame.ncols() - 1).toDoubleArray());
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        };
        return Executors.callable(runnable);
    }

    /**
     * Loads the specified dataset from the file system.
     * It uses the built-in CSV parser of Smile, provided via the
     * <code>Read</code> interface.
     * @param fileName the dataset to be loaded
     * @return a Smile DataFrame holding the CSV data
     */
    private static DataFrame readFromCsv(String fileName) throws IOException, URISyntaxException {
        String filePath = "datasets/" + fileName;
        System.out.println(">>> READ DATA FROM " + filePath);
        DataFrame dataFrame = Read.csv(filePath);
        System.out.println(dataFrame.toString(5));
        return dataFrame;
    }

    /**
     * Winsorizes a DataFrame.
     * Winsorization prevents outliers by limiting extreme feature values
     * to the 5th/95th percentile of the original distribution.
     * @param dataFrame a Smile Dataframe to be winsorized
     * @return the winsorized Dataframe; the target column (last column) remains untouched
     */
    private static DataFrame transformDataFrame(DataFrame dataFrame) {
        System.out.println(">>> TRANSFORM DATA SET");
        int targetColumnIdx = dataFrame.ncols() - 1;
        DataFrame featureDataFrame = dataFrame.drop(targetColumnIdx);
        FeatureTransform transformer = WinsorScaler.fit(featureDataFrame);
        DataFrame transformedFeatureDataFrame = transformer.transform(featureDataFrame);
        return transformedFeatureDataFrame.merge(dataFrame.column(targetColumnIdx));
    }

    /**
     * Instantiates and trains a Smile Ridge regression model.
     * The target column y of the model is the last column of the DataFrame.
     * @param dataFrame a Smile DataFrame holding training data
     * @return a fitted Smile Ridge regression model
     */
    private static LinearModel createAndFitRegressionModel(DataFrame dataFrame) {
        System.out.println(">>> FIT MODEL");
        String targetColumnName = dataFrame.names()[dataFrame.names().length - 1];
        Formula targetColumn = Formula.lhs(targetColumnName);
        return RidgeRegression.fit(targetColumn, dataFrame);
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
        System.out.println(">>> PREDICT DATA");
        return linearModel.predict(testData);
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
