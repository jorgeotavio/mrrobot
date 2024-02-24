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
		scan();
	}

	public void onHitRobot(HitRobotEvent e) {
		double turnGunAmt = normalRelativeAngleDegrees(e.getBearing() + getHeading() - getGunHeading());

		turnGunRight(turnGunAmt);
		fire(3);
	}

	public void onRoundEnded(RoundEndedEvent event) {
		Utils.writeDataToDataset();
	}
}
