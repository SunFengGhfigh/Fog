package test;

import java.util.HashMap;

import math.Matrix;
import nn.BPNN;

/**
 * 
 * @author Sun
 *
 */
public class BPNNTest {
	
	public static void main(String[] args) {
		Matrix x = new Matrix("[[0.8], [0.3]]");
		Matrix y = new Matrix("[[0.6]]");
		int[] layerDims = {2, 1};
		HashMap<String, Matrix> data = new HashMap<>();
		data.put("train_X", x);
		data.put("train_y", y);
		double learningRate = 0.5;
		int iteration = 5;
		BPNN bpnn = new BPNN(layerDims, learningRate, data);
		HashMap<String, Matrix> parameters = bpnn.initialize();
		
		Matrix W = parameters.get("W1");
		System.out.println(W);
		
		for (int i = 0; i < iteration; i++) {
			parameters = bpnn.forwardPropagation(parameters);
			parameters = bpnn.backpropagtion(parameters);
			Matrix lossM = bpnn.loss(parameters);
			System.out.println("-----------" + i + "----------");
			System.out.println("loss:" + lossM.sum());
			System.out.println("W1: " + parameters.get("W1"));
			System.out.println("dW1:" + parameters.get("dW1"));
			System.out.println("b1: " + parameters.get("b1"));
			System.out.println("db1:" + parameters.get("db1"));
			parameters = bpnn.update(parameters);
		}
	}

}
