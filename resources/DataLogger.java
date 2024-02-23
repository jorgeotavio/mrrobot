package mrrobot.resources;

import robocode.RobocodeFileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

public class DataLogger {
    private static final List<String> dataRoundLines = new ArrayList<>();
    private static final String CSV_FILE = "enemies_data.txt";
    private static final String LOG_FILE = "log.txt";

    public static double[] calculateEnemyPosition(double enemyDistance, double enemyAngle, double myX, double myY) {
        double enemyAngleRadians = Math.toRadians(enemyAngle);

        double enemyX = myX + enemyDistance * Math.sin(enemyAngleRadians);
        double enemyY = myY + enemyDistance * Math.cos(enemyAngleRadians);

        return new double[] { enemyX, enemyY };
    }

    public static void logData(double enemyDistance, double enemyAngle, double myX, double myY) {

        double[] enemyPosition = calculateEnemyPosition(enemyDistance, enemyAngle, myX, myY);

        double[] line = { enemyPosition[0], enemyPosition[1], myX, myY };
        double lastItem = line[line.length - 1];

        StringBuilder sb = new StringBuilder();

        for (double item : line) {
            
            if (item == lastItem) {
                sb.append(roundToTwoDecimalPlaces(item));
            } else {
                sb.append(roundToTwoDecimalPlaces(item) + ",");
            }
        }

        dataRoundLines.add(sb.toString());
    }

    public static void logData(String line) {
        try (RobocodeFileOutputStream rfos = new RobocodeFileOutputStream(LOG_FILE, true);
                PrintWriter pw = new PrintWriter(rfos)) {
            pw.println(line);
            pw.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeDataToFile() {
        try (RobocodeFileOutputStream rfos = new RobocodeFileOutputStream(CSV_FILE, true);
                PrintWriter pw = new PrintWriter(rfos)) {
            for (String line : dataRoundLines) {
                pw.println(line);
            }

            // when the round is finished we add "&,&,&,&" to identify it
            pw.println("&,&,&,&");

            pw.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        dataRoundLines.clear();
    }

    public static double roundToTwoDecimalPlaces(double value) {
        BigDecimal bd = new BigDecimal(Double.toString(value));
        bd = bd.setScale(2, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

}
