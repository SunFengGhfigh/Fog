package plot;

public class Test {
	
	public static void main(String[] args) {
		StdDraw.setXscale(0, 2000);
		StdDraw.setYscale(0, 2000);
		StdDraw.setPenRadius(.005);
		
		double[][] cost = {{800, 830}, {715, 840}, {900, 930},
				{830, 1240}, {1150, 1200}, {900, 1210}, 
				{1300, 1400}, {1400, 1600}, {1350, 1500},
				{1630, 1800}};
		
		for (int i = 0; i < cost.length; i++) {
			StdDraw.point(cost[i][0], cost[i][1]);
			if (i <= cost.length - 2) {
				StdDraw.line(cost[i][0], cost[i][1], cost[i + 1][0], cost[i + 1][1]);
			}
		}
		
	}

}
