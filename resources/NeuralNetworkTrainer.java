package mrrobot.resources;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;
import org.neuroph.util.TransferFunctionType;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import mrrobot.resources.DataLogger;

public class NeuralNetworkTrainer {

    private static final String DATASET_FILE = "enemies_data.txt";
    private static final int INPUT_SIZE = 4;
    private static final int OUTPUT_SIZE = 2;

    public static void train() {
        try {

            List<DataSet> matches = loadAndSeparateMatches(DATASET_FILE);

            if (!matches.isEmpty()) {
                Perceptron myPerceptron = new Perceptron(INPUT_SIZE, OUTPUT_SIZE);
                DataLogger.logData("here");

                for (DataSet match : matches) {
                    myPerceptron.learn(match);
                }
                myPerceptron.save("enemies_predictor.nnet");
            }
        } catch (Exception e) {
            StackTraceElement[] stackTraceElements = e.getStackTrace();
            if (stackTraceElements.length > 0) {
                StackTraceElement element = stackTraceElements[0];
                DataLogger.logData(e.toString() + element.getClassName() +  element.getLineNumber() );
            } else {
                System.out.println("Stack trace não disponível.");
            }
            e.printStackTrace();
        }
    }

    private static List<DataSet> loadAndSeparateMatches(String filePath) {
        List<DataSet> matches = new ArrayList<>();
        List<String> lines = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {

                if (!line.equals("&,&,&,&")) {
                    lines.add(line);
                } else {
                    if (!lines.isEmpty()) {
                        matches.add(processMatchLines(lines));
                        lines.clear();
                    }
                }
            }
            if (!lines.isEmpty()) {
                matches.add(processMatchLines(lines));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return matches;
    }

    private static DataSet processMatchLines(List<String> lines) {
        DataSet matchDataSet = new DataSet(INPUT_SIZE, OUTPUT_SIZE);
        for (int i = 0; i < lines.size() - 1; i++) {
            String[] currentLine = lines.get(i).split(",");
            String[] nextLine = lines.get(i + 1).split(",");

            if (lines.get(i).contains("&,&,&,&") || lines.get(i + 1).contains("&,&,&,&")) {
                continue;
            }

            double[] inputs = new double[INPUT_SIZE];
            for (int j = 0; j < INPUT_SIZE; j++) {
                inputs[j] = sanitizeAndConvertToDouble(currentLine[j]);
            }

            double[] outputs = new double[OUTPUT_SIZE];
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                outputs[j] = sanitizeAndConvertToDouble(nextLine[j]);
            }

            matchDataSet.addRow(new DataSetRow(inputs, outputs));
        }
        return matchDataSet;
    }

    private static double sanitizeAndConvertToDouble(String numberStr) {
        try {
            String sanitized = numberStr.replaceAll("(\\..*?)\\.", "$1");
            return Double.parseDouble(sanitized);
        } catch (NumberFormatException e) {
            System.err.println("Erro ao converter número: " + e.getMessage());
            return 0.0;
        }
    }

}
