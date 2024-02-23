package mrrobot;

import robocode.*;
import static robocode.util.Utils.normalRelativeAngleDegrees;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;

import mrrobot.resources.DataLogger;
import mrrobot.resources.NeuralNetworkTrainer;

public class MrRobot extends AdvancedRobot {
	int dist = 50;

	public void run() {
		while (true) {
			turnGunRight(5);
		}
	}

	public void onScannedRobot(ScannedRobotEvent e) {
		double enemyAngle = e.getBearing();
		double enemyDistance = e.getDistance();
		double myX = getX();
		double myY = getY();

		DataLogger.logData(enemyDistance, enemyAngle, myX, myY);
		if (enemyDistance < 50 && getEnergy() > 50) {
			fire(3);
		} else {
			fire(1);
		}

		double[] enemyPosition = DataLogger.calculateEnemyPosition(enemyDistance, enemyAngle, myX, myY);

		HttpClient client = HttpClient.newHttpClient();
		String url = String.format("http://localhost:5000/predict?enemyX=%f&enemyY=%f&myX=%f&myY=%f",
				enemyPosition[0],
				enemyPosition[1],
				myX,
				myY);

		HttpRequest request = HttpRequest.newBuilder()
				.uri(URI.create(url))
				.build();

		try {
			HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

			String[] parts = response.body().split(",");
			if (parts.length == 2) {
				double nextEnemyX = Double.parseDouble(parts[0]);
				double nextEnemyY = Double.parseDouble(parts[1]);

				// Calcula o ângulo para a posição prevista
				double angleToPredictedEnemy = absoluteBearing(myX, myY, nextEnemyX, nextEnemyY) - getGunHeading();

				// Ajusta a arma na direção do inimigo previsto
				turnGunRight(normalizeBearing(angleToPredictedEnemy));

				// Opcionalmente, atira
				fire(1);
			}

			DataLogger.logData(response.body().toString());
		} catch (IOException | InterruptedException ex) {
			System.err.println("Erro ao fazer a chamada HTTP: " + ex.getMessage());
		}

		scan();
	}

	public void onHitByBullet(HitByBulletEvent e) {
		turnRight(normalRelativeAngleDegrees(90 - (getHeading() - e.getHeading())));

		ahead(dist);
		dist *= -1;
		scan();
	}

	public void onHitRobot(HitRobotEvent e) {
		double turnGunAmt = normalRelativeAngleDegrees(e.getBearing() + getHeading() - getGunHeading());

		turnGunRight(turnGunAmt);
		fire(3);
	}

	public void onRoundEnded(RoundEndedEvent event) {
		DataLogger.writeDataToFile();
	}

	// Método para calcular o ângulo absoluto para o inimigo previsto
	double absoluteBearing(double x1, double y1, double x2, double y2) {
		double xo = x2 - x1;
		double yo = y2 - y1;
		// Calcula a hipotenusa diretamente
		double hyp = Math.sqrt(xo * xo + yo * yo);
		// Usa Math.atan2 para obter o ângulo em radianos e depois converte para graus
		double arcTan = Math.toDegrees(Math.atan2(xo, yo));
		
		// Ajusta o ângulo para que esteja no intervalo correto [0, 360)
		double bearing = arcTan;
		if (bearing < 0) {
			bearing += 360;
		}
	
		return bearing;
	}

	// Método para normalizar o ângulo
	double normalizeBearing(double angle) {
		while (angle > 180)
			angle -= 360;
		while (angle < -180)
			angle += 360;
		return angle;
	}
}
