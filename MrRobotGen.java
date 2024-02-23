package mrrobot;

import robocode.*;
import static robocode.util.Utils.normalRelativeAngleDegrees;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;

import mrrobot.resources.Utils;

public class MrRobotGen extends AdvancedRobot {
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
	
		Utils.logData(enemyDistance, enemyAngle, myX, myY);
		if (enemyDistance < 50 && getEnergy() > 50) {
			fire(3);
		} else {
			fire(1);
		}
		
		scan();
	}

	public void onHitByBullet(HitByBulletEvent e) {
		turnRight(normalRelativeAngleDegrees(90 - (getHeading() - e.getHeading())));

		ahead(dist);
		dist *= -1;
		// scan();
	}

	public void onHitRobot(HitRobotEvent e) {
		double turnGunAmt = normalRelativeAngleDegrees(e.getBearing() + getHeading() - getGunHeading());

		turnGunRight(turnGunAmt);
		fire(3);
	}

	public void onRoundEnded(RoundEndedEvent event) {
		Utils.writeDataToDataset();
	}

	// Metodo para calcular o ângulo absoluto para o inimigo previsto
	double absoluteBearing(double x1, double y1, double x2, double y2) {
		double xo = x2 - x1;
		double yo = y2 - y1;
		// valcula a hipotenusa diretamente
		double hyp = Math.sqrt(xo * xo + yo * yo);
		// usa Math.atan2 para obter o ângulo em radianos e depois converte para graus
		double arcTan = Math.toDegrees(Math.atan2(xo, yo));
		
		// ajusa o angulo para que esteja no intervalo correto [0, 360)
		double bearing = arcTan;
		if (bearing < 0) {
			bearing += 360;
		}
	
		return bearing;
	}

	// Metodo para normalizar o ângulo
	double normalizeBearing(double angle) {
		while (angle > 180)
			angle -= 360;
		while (angle < -180)
			angle += 360;
		return angle;
	}
}
