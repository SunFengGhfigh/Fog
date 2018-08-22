package plot;

/*************************************************************************
 * 
 * @author Yuyan Wang
 * @date 2018.8
 * 
 * The secondary packaging of the graph display,
 * including the loss function point diagram etc.
 *
 *************************************************************************/

import math.Matrix;

public class Plt {
	
	/**
	 * Matrix: [0]: iteration [1]: cost or loss.
	 * Return a graph of cost in training of neural network.
	 * @param m
	 */
	public void plot(Matrix m) {
		double[][] array = m.getArray();
		double minX = array[0][0], maxX = array[0][0], minY = array[0][1], maxY = array[0][1];
		
		for (int i = 1; i < array.length; i++) {
			if (array[i][0] < minX) {
				minX = array[i][0];
			}
			if (array[i][0] > maxX) {
				maxX = array[i][0];
			}
			if (array[i][1] < minY) {
				minY = array[i][1];
			}
			if (array[i][1] > maxY) {
				maxY = array[i][1];
			}
		}
		
		StdDraw.setXscale(minX, maxX);
		StdDraw.setYscale(minY, maxY);
		StdDraw.setPenRadius(.005);
		
		for (int i = 0; i < array.length; i++) {
			StdDraw.point(array[i][0], array[i][1]);
			System.out.println(array[i][0] + " " + array[i][1]);
			if (i <= array.length - 2) {
				StdDraw.line(array[i][0], array[i][1], array[i + 1][0], array[i + 1][1]);
			}
		}
	}

}
