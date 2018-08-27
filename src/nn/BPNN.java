package nn;

import java.io.File;

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
	 * Loss function.
	 * @param parameters
	 * @return
	 */
	private double loss(HashMap<String, Matrix> parameters) {
		Matrix y_hat = parameters.get(("A" + (layerDims.length - 1)));
		Matrix y = data.get("train_y");
		Matrix result = Expansion.loss(y_hat, y);
		double sum = result.sum();
		if (Const.L2regularization) {
			sum += Expansion.L2regularization(layerDims, parameters, Const.lambda);
		}
		return sum;
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
		Matrix train_y = data.get("train_y");
		parameters = Expansion.backwardPropagation(parameters, layerDims, train_y);
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
				double cost = loss(parameters);
				System.out.println("------------" + i + "------------");
				System.out.println("loss:" + cost);
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
	 * Choose cross entropy loss function.
	 */
	public void cross_entropy() {
		Const.crossEntropy = true;
	}
	
	/**
	 * Choose L2 loss function.
	 */
	public void L2() {
		Const.L2 = true;
	}
	
	/**
	 * Choose L1 loss function.
	 */
	public void L1() {
		Const.L1 = true;
	}
	
	/**
	 * Choose L2 regularization.
	 */
	public void L2Regularization(double lambda) {
		Const.L2regularization = true;
		Const.lambda = lambda;
	}
	
	/**
	 * Choose dropout regularization.
	 */
	public void dropout() {
		Const.dropout = true;
	}
	
	/*************************************************************************
	 *  Save model or read model.
	 *************************************************************************/
	public void saveModel(String pathname) {
		File file = new File(pathname);
		file.mkdirs();
		// 序列化
	}
	
	public BPNN readmodel() {
		return null;
	}
	
	
}
