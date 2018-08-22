package nn;

/*************************************************************************
 * 
 * @author Sun
 * @author Yuyan Wang
 * @date 2018.8
 * 
 * The library of back propagation neural network.
 * 
 *************************************************************************/

import java.util.HashMap;

import math.Expansion;
import math.Matrix;

public class BPNN {
	
	private int[] layerDims;
	private double learningRate;
	private HashMap<String, Matrix> data;
	
	public BPNN(int[] layerDims, double learningRate, HashMap<String, Matrix> data) {
		this.layerDims = layerDims;
		this.learningRate = learningRate;
		this.data = data;
	}
	
	public HashMap<String, Matrix> initialize(){
		HashMap<String, Matrix> parameters = new HashMap<>(4 * layerDims.length);
		Matrix train_X = data.get("train_X");
		parameters.put("A0", train_X);
		for (int i = 1; i < layerDims.length; i++) {
			Matrix W = new Matrix(layerDims[i], layerDims[i - 1], "random");
			Matrix b = new Matrix(layerDims[i], 1);
			parameters.put("W" + i, W);
			parameters.put("b" + i, b);
		}
		return parameters;
	}
	
	public HashMap<String, Matrix> forwardPropagation(HashMap<String, Matrix> parameters) {
		for (int i = 1; i < layerDims.length; i++) {
			Matrix W = parameters.get("W" + i);
			Matrix b = parameters.get("b" + i);
			Matrix prevA = parameters.get("A" + (i - 1));
			Matrix Z = W.dot(prevA).add(b);
			Matrix A = null;
			if (i < layerDims.length - 1) {
				A = Expansion.relu(Z);
			} else {
				A = Expansion.sigmoid(Z);
			}
			parameters.put("Z" + i, Z);
			parameters.put("A" + i, A);
		}
		return parameters;
	}
	
	public Matrix loss(HashMap<String, Matrix> parameters) {
		Matrix y_hat = parameters.get(("A" + (layerDims.length - 1)));
		Matrix y = data.get("train_y");
		Matrix result = Expansion.crossEntropy(y_hat, y);
		return result;
	}
	
	public HashMap<String, Matrix> backpropagtion(HashMap<String, Matrix> parameters){
		Matrix train_X = data.get("train_X");
		Matrix train_y = data.get("train_y");
		int m = train_X.width;
		
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
	
	public HashMap<String, Matrix> update(HashMap<String, Matrix> parameters){
		for (int i = 1; i < layerDims.length; i++) {
			Matrix W = parameters.get("W" + i);
			Matrix b = parameters.get("b" + i);
			Matrix dW = parameters.get("dW" + i);
			Matrix db = parameters.get("db" + i);
			parameters.put("W" + i, dW.dot(-1.0d * learningRate).add(W));
			parameters.put("b" + i, db.dot(-1.0d * learningRate).add(b));
		}
		return parameters;
	}
	
}
