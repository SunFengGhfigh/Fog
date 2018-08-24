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

import java.util.HashMap;

import nn.Const;

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
	
	public static double sigmoid(double a) {
		return 1.0d / (1.0d + Math.exp(-a));
	}
	
	/**
	 * The derivation of sigmoid function.
	 * @param a
	 * @return
	 */
	public static Matrix dSigmoid(Matrix a) {
		Matrix k = sigmoid(a);
		return k.dot(k.dot(-1).add(1));
	}
	
	public static double dSigmoid(double a) {
		double k = sigmoid(a);
		return k * (1.0d - k);
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
	 * Loss function.
	 * @param y_hat
	 * @param y
	 * @return
	 */
	public static Matrix loss(Matrix y_hat, Matrix y) {
		if (Const.crossEntropy) {
			return crossEntropy(y_hat, y);
		} else if (Const.L2) {
			return L2(y_hat, y);
		} else if (Const.L1) {
			return L1(y_hat, y);
		}
		return crossEntropy(y_hat, y);
	}
	
	/**
	 * The cross entropy loss function.
	 * Formula: L(y_hat, y) = y*log(y_hat) + (1-y)*log(1-h_hat)
	 * @param y_hat
	 * @param y
	 * @return
	 */
	private static Matrix crossEntropy(Matrix y_hat, Matrix y) {
		int m = y.width;
		Matrix temp1 = y.multiply(y_hat.log());
		Matrix temp2_1 = y.dot(-1).add(1);
		Matrix temp2_2 = y_hat.dot(-1).add(1).log();
		Matrix temp2 = temp2_1.multiply(temp2_2);
		Matrix temp3 = temp1.add(temp2);
		Matrix result = temp3.dot(-1.0d / m);
		return result;
	}
	
	/**
	 * The L2 loss function.
	 * @param y_hat
	 * @param y
	 * @return
	 */
	private static Matrix L2(Matrix y_hat, Matrix y) {
		Matrix temp1 = y.add(y_hat.dot(-1));
		Matrix temp2 = temp1.multiply(temp1);
		Matrix result = temp2.dot(0.5d);
		return result;
	}
	
	/**
	 * The L1 loss function.
	 * @param y_hat
	 * @param y
	 * @return
	 */
	private static Matrix L1(Matrix y_hat, Matrix y) {
		Matrix temp1 = y.add(y_hat.dot(-1));
		double[][] array = temp1.getArray();
		for (int i = 0; i < temp1.height; i++) {
			for (int j = 0; j < temp1.width; j++) {
				if (array[i][j] < 0) {
					array[i][j] = Math.abs(array[i][j]);
				}
			}
		}
		temp1.setArray(array);
		return temp1;
	}
	
	/*************************************************************************
	 *  The derivation of Loss function.
	 *************************************************************************/
	
	/**
	 * 
	 * @param parameters
	 * @param layerDims
	 * @param train_y
	 * @return
	 */
	public static HashMap<String, Matrix> backwardPropagation(HashMap<String, Matrix> parameters, int[] layerDims, Matrix train_y){
		if (Const.crossEntropy) {
			return crossEntropyBack(parameters, layerDims, train_y);
		} else if (Const.L2) {
			return L2Back(parameters, layerDims, train_y);
		} else if (Const.L1) {
			return L1Back(parameters, layerDims, train_y);
		}
		return crossEntropyBack(parameters, layerDims, train_y);
	}
	
	/**
	 * The derivation process of cross entropy.
	 * @param parameters
	 * @param layerDims
	 * @param train_y
	 * @return
	 */
	private static HashMap<String, Matrix> crossEntropyBack(HashMap<String, Matrix> parameters, int[] layerDims, Matrix train_y){
		int m = train_y.width;
		for (int i = layerDims.length - 1; i > 0; i--) {
			Matrix dZ, dW, db;
			if (i == layerDims.length - 1) {
				Matrix temp = train_y.dot(-1);
				Matrix A = parameters.get("A" + i);
				dZ = A.add(temp);
				Matrix prevA = parameters.get("A" + (i - 1));
				dW = dZ.dot(prevA.T()).dot(1.0d/m);
				db = dZ.sum(1).dot(1.0d/m);
			} else {
				Matrix W = parameters.get("W" + (i + 1));
				Matrix prevdZ = parameters.get("dZ" + (i + 1));
				Matrix dA = W.T().dot(prevdZ);
				dZ = Expansion.relu(dA);
				Matrix prevA = parameters.get("A" + (i - 1));
				dW = dZ.dot(prevA.T()).dot(1.0d/m);
				db = dZ.sum(1).dot(1.0d/m);
			}
			parameters.put("dZ" + i, dZ);
			parameters.put("dW" + i, dW);
			parameters.put("db" + i, db);
		}
		return parameters;
	}
	
	/**
	 * The derivation process of L2.
	 * L2 = 1/2 * (y - y_hat) * (y - y_hat)
	 * 
	 * @param parameters
	 * @param layerDims
	 * @param train_y
	 * @return
	 */
	private static HashMap<String, Matrix> L2Back(HashMap<String, Matrix> parameters, int[] layerDims, Matrix train_y){
		int m = train_y.width;
		for (int i = layerDims.length - 1; i > 0; i--) {
			Matrix dZ, dW, db;
			if (i == layerDims.length - 1) {
				Matrix A = parameters.get("A" + i);
				Matrix temp1 = A.multiply(A);
				Matrix temp2 = temp1.multiply(A).dot(-1);
				Matrix temp3 = train_y.multiply(A).dot(-1);
				Matrix temp4 = train_y.multiply(temp1);
				dZ = temp1.add(temp2).add(temp3).add(temp4);
				Matrix prevA = parameters.get("A" + (i - 1));
				dW = dZ.dot(prevA.T()).dot(1.0d/m);
				db = dZ.sum(1).dot(1.0d/m);
			} else {
				Matrix W = parameters.get("W" + (i + 1));
				Matrix prevdZ = parameters.get("dZ" + (i + 1));
				Matrix dA = W.T().dot(prevdZ);
				dZ = Expansion.relu(dA);
				Matrix prevA = parameters.get("A" + (i - 1));
				dW = dZ.dot(prevA.T()).dot(1.0d/m);
				db = dZ.sum(1).dot(1.0d/m);
			}
			parameters.put("dZ" + i, dZ);
			parameters.put("dW" + i, dW);
			parameters.put("db" + i, db);
		}
		return parameters;
	}
	
	/**
	 * The derivation process of L1.
	 * L1 = |y-y_hat|
	 * derivative of y_hat:
	 * y - y_hat >= 0 -> -1
	 * y - y_hat < 0  ->  1
	 * 
	 * @param parameters
	 * @param layerDims
	 * @param train_y
	 * @return
	 */
	private static HashMap<String, Matrix> L1Back(HashMap<String, Matrix> parameters, int[] layerDims, Matrix train_y){
		int m = train_y.width;
		for (int i = layerDims.length - 1; i > 0; i--) {
			Matrix dZ, dW, db;
			if (i == layerDims.length - 1) {
				Matrix A = parameters.get("A" + i);
				Matrix Z = parameters.get("Z" + i);
				Matrix temp1 = train_y.add(A.dot(-1));
				double[][] array = temp1.getArray();
				double[][] zArray = Z.getArray();
				for (int p = 0; p < temp1.height; p++) {
					for (int q = 0; q < temp1.width; q++) {
						if (array[p][q] >= 0) {
							array[p][q] = -dSigmoid(zArray[p][q]);
						} else {
							array[p][q] = dSigmoid(zArray[p][q]);
						}
					}
				}
				temp1.setArray(array);
				dZ = temp1;
				Matrix prevA = parameters.get("A" + (i - 1));
				dW = dZ.dot(prevA.T()).dot(1.0d/m);
				db = dZ.sum(1).dot(1.0d/m);
			} else {
				Matrix W = parameters.get("W" + (i + 1));
				Matrix prevdZ = parameters.get("dZ" + (i + 1));
				Matrix dA = W.T().dot(prevdZ);
				dZ = Expansion.relu(dA);
				Matrix prevA = parameters.get("A" + (i - 1));
				dW = dZ.dot(prevA.T()).dot(1.0d/m);
				db = dZ.sum(1).dot(1.0d/m);
			}
			parameters.put("dZ" + i, dZ);
			parameters.put("dW" + i, dW);
			parameters.put("db" + i, db);
		}
		return parameters;
	}

}
