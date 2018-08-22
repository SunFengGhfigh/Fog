package math;

/*************************************************************************
 * 
 * @author Yuyan Wang
 * @date 2018.8
 * 
 * The extended matrix calculation contains the computational 
 * functions of most neural networks.
 *
 *************************************************************************/

public class Expansion{
	
	/*************************************************************************
	 *  Activation function.
	 *************************************************************************/
	
	/**
	 * Rectified Linear Unit.
	 * Example: K = [[-1, 2, 3]] P = ReLU(K)
	 * P = [[0, 2, 3]]
	 * @param a
	 * @return
	 */
	public static Matrix relu(Matrix a) {
		Matrix temp = a.clone();
		double[][] array = temp.getArray();
		for (int i = 0; i < temp.height; i++) {
			for (int j = 0; j < temp.width; j++) {
				if (array[i][j] < 0) {
					array[i][j] = 0;
				}
			}
		}
		temp.setArray(array);
		return temp;
	}
	
	/**
	 * Sigmoid function.
	 * @param a
	 * @return
	 */
	public static Matrix sigmoid(Matrix a) {
		Matrix temp = a.clone();
		double[][] array = temp.getArray();
		for (int i = 0; i < temp.height; i++) {
			for (int j = 0; j < temp.width; j++) {
				array[i][j] = 1.0d / (1.0d + Math.exp(-array[i][j]));
			}
		}
		return temp;
	}
	
	/*************************************************************************
	 *  Loss function.
	 *************************************************************************/
	
	public static Matrix crossEntropy(Matrix y_hat, Matrix y) {
		int m = y.width;
		Matrix temp1 = y.multiply(y_hat.log());
		Matrix temp2_1 = y.dot(-1).add(1);
		Matrix temp2_2 = y_hat.dot(-1).add(1).log();
		Matrix temp2 = temp2_1.multiply(temp2_2);
		Matrix temp3 = temp1.add(temp2);
		Matrix result = temp3.dot(-1.0d / m);
		return result;
	}

}
