package test;

import math.Expansion;
import math.Matrix;

/**
 * 
 * @author Sun
 *
 */
public class MatrixTest {
	
	public static void main(String[] args) {
		Matrix a = new Matrix("[[2, 3, 4]]");
		System.out.println(Expansion.sigmoid(a));
	}

}
