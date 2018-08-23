package nn;

/*************************************************************************
 * 
 * @author Sun
 * @date 2018.8
 * 
 * The library of back propagation neural network.
 * 
 *************************************************************************/

import java.util.HashMap;

import math.Expansion;
import math.Matrix;

public class BPNN {
	
	// Layer dimension
	private int[] layerDims;
	private double learningRate;
	private HashMap<String, Matrix> data;
	private int iteration;
	private int printSize;
	private int printBatch = 10;
	
	/**
	 * Construct method.
	 * @param layerDims
	 * @param learningRate
	 * @param data
	 * @param iteration
	 */
	public BPNN(int[] layerDims, double learningRate, HashMap<String, Matrix> data, int iteration) {
		this.layerDims = layerDims;
		this.learningRate = learningRate;
		this.data = data;
		this.iteration = iteration;
		if (iteration > 20) {
			this.printSize = iteration / printBatch;
		} else {
			this.printSize = 1;
		}
	}
	
	/**
	 * Initialize parameters include weights and biases.
	 * When initialize the weights, the default method is random assignment.
	 * The default value of weight rate is 1e-2.
	 * We also provide other initialization methods, such as how the Kaiming He team proposed the initialization method (He)
	 * and Xavier.
	 * In practical engineering, we usually prefer He's initialization method because of it's limitations on the variance
	 * of weights. And that is precisely the flaw of Xavier.
	 * @return
	 */
	private HashMap<String, Matrix> initialize(){
		HashMap<String, Matrix> parameters = new HashMap<>(4 * layerDims.length);
		Matrix train_X = data.get("train_X");
		parameters.put("A0", train_X);
		parameters = Expansion.initialize(parameters, layerDims);
		return parameters;
	}
	
	/**
	 * Forward propagation.
	 * @param parameters
	 * @return
	 */
	private HashMap<String, Matrix> forwardPropagation(HashMap<String, Matrix> parameters) {
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
	
	/**
	 * 
	 * @param parameters
	 * @return
	 */
	private Matrix loss(HashMap<String, Matrix> parameters) {
		Matrix y_hat = parameters.get(("A" + (layerDims.length - 1)));
		Matrix y = data.get("train_y");
		Matrix result = Expansion.crossEntropy(y_hat, y);
		return result;
	}
	
	/**
	 * Back propagation.
	 * This is just one example: ReLU -> Sigmoid -> Cross Entropy.
	 * So, we haven't finished the automatic derivation part yet, but the demo is done and ready
	 * to run. We're working on the details.
	 * @param parameters
	 * @return
	 */
	private HashMap<String, Matrix> backpropagtion(HashMap<String, Matrix> parameters){
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
	
	/**
	 * Update weights and biases.
	 * @param parameters
	 * @return
	 */
	private HashMap<String, Matrix> update(HashMap<String, Matrix> parameters){
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
	
	/**
	 * Train the model.
	 * @return
	 */
	public HashMap<String, Matrix> train() {
		HashMap<String, Matrix> parameters = initialize();
		
		for (int i = 0; i < this.iteration; i++) {
			parameters = forwardPropagation(parameters);
			if (i % this.printSize == 0) {
				Matrix cost = loss(parameters);
				System.out.println("------------" + i + "------------");
				System.out.println("loss:" + cost.sum());
			}
			parameters = backpropagtion(parameters);
			parameters = update(parameters);
		}
		
		return parameters;
	}
	
	/**
	 * Predict the train data.
	 * @param parameters
	 */
	public void predictTrain(HashMap<String, Matrix> parameters) {
		parameters = forwardPropagation(parameters);
		int L = layerDims.length;
		Matrix y_hat = parameters.get("A" + (L - 1));
		System.out.println(y_hat);
	}
	
	/**
	 * Predict the validation data.
	 * @param parameters
	 */
	public void predictValidation(HashMap<String, Matrix> parameters) {
		parameters.put("A0", data.get("val_X"));
		parameters = forwardPropagation(parameters);
		int L = layerDims.length;
		Matrix y_hat = parameters.get("A" + (L - 1));
		System.out.println(y_hat);
	}
	
	/*************************************************************************
	 *  Configure.
	 *************************************************************************/
	
	/**
	 * Choose the random initialization parameter.
	 * The default value of weightRate is 1e-2.
	 * @param weightRate
	 */
	public void initializeRandom(double weightRate) {
		Const.random = true;
		Const.weightRate = weightRate;
	}
	
	/**
	 * Choose He's method initialization parameter.
	 */
	public void He() {
		Const.He = true;
	}
	
	/**
	 * Choose Xavier method initialization parameter.
	 */
	public void Xavier() {
		Const.Xavier = true;
	}
	
	/**
	 * Select tanh as the activation function.
	 */
	public void tanh() {
		Const.tanh = true;
	}
	
}
