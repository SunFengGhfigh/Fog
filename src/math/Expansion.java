package math;

import java.util.HashMap;

import nn.Const;

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
	 *  Initialize method.
	 *************************************************************************/
	public static HashMap<String, Matrix> initialize(HashMap<String, Matrix> parameters, int[] layerDims){
		if (Const.He) {
			return He(parameters, layerDims);
		}
		if (Const.Xavier) {
			return Xavier(parameters, layerDims);
		}
		return random(parameters, layerDims);
	}
	
	/**
	 * W = 0.01 * randn(n_in, n_out)
	 * @param parameters
	 * @param layerDims
	 * @return
	 */
	private static HashMap<String, Matrix> random(HashMap<String, Matrix> parameters, int[] layerDims){
		for (int i = 1; i < layerDims.length; i++) {
			Matrix W = new Matrix(layerDims[i], layerDims[i - 1], "random").dot(Const.weightRate);
			Matrix b = new Matrix(layerDims[i], 1);
			parameters.put("W" + i, W);
			parameters.put("b" + i, b);
		}
		return parameters;
	}
	
	/**
	 * W = 0.01 * randn(n_in, n_out) / sqrt(n_in)
	 * @param parameters
	 * @param layerDims
	 * @return
	 */
	private static HashMap<String, Matrix> Xavier(HashMap<String, Matrix> parameters, int[] layerDims){
		for (int i = 1; i < layerDims.length; i++) {
			Matrix W = new Matrix(layerDims[i], layerDims[i - 1], "random").dot(Const.weightRate).dot(1.0d/Math.sqrt(layerDims[i]));
			Matrix b = new Matrix(layerDims[i], 1);
			parameters.put("W" + i, W);
			parameters.put("b" + i, b);
		}
		return parameters;
	}
	
	/**
	 * W = 0.01 * randn(n_in, n_out) / sqrt(n_in / 2)
	 * @param parameters
	 * @param layerDims
	 * @return
	 */
	private static HashMap<String, Matrix> He(HashMap<String, Matrix> parameters, int[] layerDims){
		for (int i = 1; i < layerDims.length; i++) {
			Matrix W = new Matrix(layerDims[i], layerDims[i - 1], "random").dot(Const.weightRate).dot(1.0d/(Math.sqrt(layerDims[i] / 2.0d)));
			Matrix b = new Matrix(layerDims[i], 1);
			parameters.put("W" + i, W);
			parameters.put("b" + i, b);
		}
		return parameters;
	}
	
	
	/*************************************************************************
	 *  Activation function.
	 *************************************************************************/
	
	/**
	 * Rectified Linear Unit.
	 * Example: K = [[-1, 2, 3]] P = ReLU(K)
	 * P = [[0, 2, 3]]
	 * 
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
	
	/**
	 * tanh(x) = 2sigmoid(2x)-1
	 * @param a
	 * @return
	 */
	public static Matrix tanh(Matrix a) {
		Matrix temp1 = a.dot(2);
		Matrix temp2 = sigmoid(temp1).dot(2);
		return temp2.add(-1);
	}
	
	/*************************************************************************
	 *  Loss function.
	 *************************************************************************/
	
	/**
	 * The cross entropy loss function.
	 * Formula: L(y_hat, y) = y*log(y_hat) + (1-y)*log(1-h_hat)
	 * @param y_hat
	 * @param y
	 * @return
	 */
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
