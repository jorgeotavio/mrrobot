package mrrobot;

import robocode.*;
import static robocode.util.Utils.normalRelativeAngleDegrees;

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

	public void onBattleEnded(BattleEndedEvent event) {
		NeuralNetworkTrainer.train();
	}
}

