package math;

/*************************************************************************
 * 
 * @author Sun
 * @author Yuyan Wang
 * @date 2018.8
 * 
 * Basic operations of matrix. Standard matrix library.
 * 
 *
 *************************************************************************/

import java.util.Arrays;
import java.util.Random;

public class Matrix {
	
	public int height;
	public int width;
	public String shape;
	private double[][] array;
	
	/**
	 * Construct method.
	 * If the configure is 'random', it will return a matrix with a gaussian distribution.
	 * If the configure is 'like sn', it will return a matrix with specify number.
	 * 
	 * @param height
	 * @param width
	 * @param type
	 */
	public Matrix(int height, int width, String...type) {
		this.height = height;
		this.width = width;
		this.array = new double[height][width];
		this.shape = shape();
		int typeLength = type.length;
		
		if (typeLength == 1) {
			if (type[0].equals("random")) {
				randomArray();
			} else if (type[0].indexOf("like") > -1) {
				specifyArray(type[0]);
			}
		} else if (typeLength != 0){
			System.err.println("Initialize matrix ERROR: Instruction is invalid! -> " + Arrays.toString(type));
		}
	}
	
	/**
	 * Construct method.
	 * Generate the matrix according to the configuration.
	 * 
	 * @param config
	 */
	public Matrix(String config) {
		config = config.replaceAll(" ", "");
		config = config.substring(1, config.length() - 1);
		String[] content = config.split("\\],\\[");
		this.width = content[0].replaceAll("\\[", "").replaceAll("\\]", "").split(",").length;
		this.height = content.length;
		this.array = new double[this.height][this.width];
		this.shape = shape();
		
		for (int i = 0; i < this.height; i++) {
			String[] temp = content[i].replaceAll("\\[", "").replaceAll("\\]", "").split(",");
			if (temp.length != this.width) {
				System.err.println("Initialize matrix ERROR: Initialization matrix failed.");
			}
			for (int j = 0; j < this.width; j++) {
				this.array[i][j] = Float.parseFloat(temp[j]);
			}
		}
	}
	
	/**
	 * Get a random array with gaussian distribution.
	 */
	private void randomArray(){
		Random random = new Random();
		for (int i = 0; i < this.height; i++) {
			for (int j = 0; j < this.width; j++) {
				this.array[i][j] = random.nextGaussian();
			}
		}
	}
	
	/**
	 * Get a array with specify number.
	 * Example: matrix k = new Matrix(1, 1, "like 19")
	 * k is [[19.0]]
	 */
	private void specifyArray(String str) {
		String[] strs = str.split(" ");
		double specifyNumber = Double.parseDouble(strs[1]);
		for (int i = 0; i < this.height; i++) {
			for (int j = 0; j < this.width; j++) {
				this.array[i][j] = specifyNumber;
			}
		}
	}
	
	
	/*************************************************************************
	 *  Scientific computing.
	 *************************************************************************/
	
	/**
	 * The sum of matrix.
	 * 
	 * @return
	 */
	public double sum() {
		double sum = 0;
		for (int i = 0; i < this.height; i++) {
			for (int j = 0; j < this.width; j++) {
				sum += this.array[i][j];
			}
		}
		return sum;
	}
	
	public Matrix sum(int axis) {
		if (axis == 0) {
			Matrix temp = new Matrix(1, this.width);
			for (int i = 0; i < this.width; i++) {
				for (int j = 0; j < this.height; j++) {
					temp.array[0][i] += this.array[j][i];
				}
			}
			return temp;
		} else {
			Matrix temp = new Matrix(this.height, 1);
			for (int i = 0; i < this.height; i++) {
				for (int j = 0; j < this.width; j++) {
					temp.array[i][0] += this.array[i][j];
				}
			}
			return temp;
		}
	}
	
	/**
	 * The average of matrix.
	 * 
	 * @return
	 */
	public double avg() {
		double sum = sum();
		double avg = sum / (this.height * this.width);
		return avg;
	}
	
	/**
	 * Matrix logarithm.
	 * 
	 * @return
	 */
	public Matrix log() {
		Matrix temp = this.clone();
		for (int i = 0; i < temp.height; i++) {
			for (int j = 0; j < temp.width; j++) {
				temp.array[i][j] = Math.log(temp.array[i][j]);
			}
		}
		return temp;
	}
	
	/**
	 * Matrix add a number.
	 * 
	 * @param number
	 * @return a new matrix.
	 */
	public Matrix add(double number) {
		Matrix temp = this.clone();
		for (int i = 0; i < temp.height; i++) {
			for (int j = 0; j < temp.width; j++) {
				temp.array[i][j] += number;
			}
		}
		return temp;
	}
	
	public Matrix add(Matrix m) {
		if (this.height == m.height && this.width == m.width) {
			Matrix temp = this.clone();
			for (int i = 0; i < this.height; i++) {
				for (int j = 0; j < this.width; j++) {
					temp.array[i][j] += m.array[i][j];
				}
			}
			return temp;
		} else {
			Matrix[] list = broadcastForAdd(this, m);
			return list[0].add(list[1]);
		}
	}
	
	/**
	 * Matrix multiplication.
	 * @param m
	 * @return
	 */
	public Matrix multiply(Matrix m) {
		if (this.height == m.height && this.width == m.width) {
			Matrix temp = this.clone();
			for (int i = 0; i < this.height; i++) {
				for (int j = 0; j < this.width; j++) {
					temp.array[i][j] *= m.array[i][j];
				}
			}
			return temp;
		} else {
			Matrix[] list = broadcastForAdd(this, m);
			return list[0].multiply(list[1]);
		}
	}
	
	/**
	 * Broadcast mechanism in matrix addition.
	 * Example:[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] + [[100.0], [999.0]]
	 * Result: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] + [[100.0, 100.0, 100.0], [999.0, 999.0, 999.0]]
	 * @param M
	 * @param N
	 * @return
	 */
	private Matrix[] broadcastForAdd(Matrix M, Matrix N) {
		Matrix[] matrixList = null;
		Matrix a = M.clone(), b = N.clone();
		int aH = a.height, aW = a.width, bH = b.height, bW = b.width;
		
		boolean flag = false;
		if (aH >= bH && aW >= bW) {
			if (aH % bH == 0 && aW % bW == 0) {
				Matrix temp = new Matrix(aH, aW, "zeros");
				for (int i = 0; i < aH; i++) {
					for (int j = 0; j < aW; j++) {
						temp.array[i][j] = b.array[i % bH][j % bW];
					}
				}
				matrixList = new Matrix[2];
				matrixList[0] = a;
				matrixList[1] = temp;
				flag = true;
			}
		} else if (aH <= bH && aW <= bW) {
			if (bH % aH == 0 && bW % aW == 0) {
				Matrix temp = new Matrix(bH, bW, "zeros");
				for (int i = 0; i < bH; i++) {
					for (int j = 0; j < bW; j++) {
						temp.array[i][j] = a.array[i % aH][j % aW];
					}
				}
				matrixList = new Matrix[2];
				matrixList[0] = temp;
				matrixList[1] = b;
				flag = true;
			}
		}
		if (!flag) {
			System.err.println("Broadcast fro shape ERROR: Can't broadcast, dimensions are don't match");
		}
		return matrixList;
	}
	
	/**
	 * Dot product of matrix.
	 * 
	 * @param number
	 * @return a new matrix.
	 */
	public Matrix dot(double number) {
		Matrix temp = this.clone();
		for (int i = 0; i < temp.height; i++) {
			for (int j = 0; j < temp.width; j++) {
				temp.array[i][j] *= number;
			}
		}
		return temp;
	}
	
	public Matrix dot(Matrix m) {
		Matrix temp = this.clone();
		double[][] result = new double[this.height][m.width];
		for (int i = 0; i < temp.height; i++) {
			for (int j = 0; j < m.width; j++) {
				Matrix tempA = temp.sub(i + ":" + i + ", :");
				Matrix tempB = m.sub(":, " + j + ":" + j);
				result[i][j] = opMultiOneLine(tempA, tempB);
			}
		}
		Matrix resultMatrix = new Matrix(temp.height, m.width);
		resultMatrix.array = result;
		return resultMatrix;
	}
	
	private double opMultiOneLine(Matrix a, Matrix b) {
		double sum = 0f;
		int aHeight = a.height, aWidth = a.width, bHeight = b.height, bWidth = b.width;
		if (aHeight == 1 && bWidth == 1) {
			if (aWidth == bHeight) {
				for (int i = 0; i < aWidth; i++) {
					sum += (a.array[0][i] * b.array[i][0]);
				}
			} else {
				System.out.println("opMultiOrDivOneLine Error:乘法或除法操作错误，维度不匹配");
			}
		}
		return sum;
	}
	
	
	/*************************************************************************
	 *  Matrix representation.
	 *************************************************************************/
	
	/**
	 * Get the shape of matrix.
	 * Example: matrix k is [[1, 2, 3], [4, 5, 6]]
	 * The shape of k is [2, 3]
	 * 
	 * @return String of shape
	 */
	private String shape() {
		String str = "[";
		str += this.height + ", " + this.width;
		str += "]";
		return str;
	}
	
	/**
	 * Return a matrix same as this.
	 */
	public Matrix clone() {
		Matrix clone = new Matrix(this.height, this.width);
		for (int i = 0; i < clone.height; i++) {
			for (int j = 0; j < clone.width; j++) {
				clone.array[i][j] = this.array[i][j];
			}
		}
		return clone;
	}
	
	/**
	 * The sub view of matrix.
	 * Example:K = [[1, 2, 3], [4, 5, 6]]
	 * P = K.sub("1:, 1:")
	 * P = [5, 6]
	 * @param str
	 * @return
	 */
	public Matrix sub(String str) {
		String[] config = null;
		int heightStart = 0, heightEnd = this.height - 1, widthStart = 0, widthEnd = this.width - 1;
		
		if (str.indexOf(", ") > -1) {
			config = str.split(", ");
		} else {
			config = str.split(",");
		}
		if (!config[0].equals(":")) {
			if (config[0].indexOf(":") == config[0].length() - 1) {
				heightStart = Integer.parseInt(config[0].split(":")[0]);
			} else if (config[0].indexOf(":") == 0) {
				heightEnd = Integer.parseInt(config[0].split(":")[1]);
			} else {
				heightStart = Integer.parseInt(config[0].split(":")[0]);
				heightEnd = Integer.parseInt(config[0].split(":")[1]);
			}
		}
		if (!config[1].equals(":")) {
			if (config[1].indexOf(":") == config[1].length() - 1) {
				widthStart = Integer.parseInt(config[1].split(":")[0]);
			} else if (config[1].indexOf(":") == 0) {
				widthEnd = Integer.parseInt(config[1].split(":")[1]);
			} else {
				widthStart = Integer.parseInt(config[1].split(":")[0]);
				widthEnd = Integer.parseInt(config[1].split(":")[1]);
			}
		}
		int newHeight = heightEnd - heightStart + 1;
		int newWidth = widthEnd - widthStart + 1;
		
		Matrix m = new Matrix(newHeight, newWidth, "zeros");
		for (int i = 0; i < newHeight; i++) {
			for (int j = 0; j < newWidth; j++) {
				m.array[i][j] = this.array[heightStart + i][widthStart + j];
			}
		}
		return m;
	}
	
	/**
	 * Transposed matrix.
	 * @return
	 */
	public Matrix T() {
		Matrix result = new Matrix(this.width, this.height);
		for (int i = 0; i < result.height; i++) {
			for (int j = 0; j < result.width; j++) {
				result.array[i][j] = this.array[j][i];
			}
		}
		return result;
	}
	
	/**
	 * Get the array.
	 * @return
	 */
	public double[][] getArray(){
		return this.array;
	}
	
	/**
	 * Set the array.
	 * @param array
	 */
	public void setArray(double[][] array) {
		this.array = array;
	}
	
	@Override
	public String toString() {
		String str = "array([";
		for (int i = 0; i < this.height; i++) {
			for (int j = 0; j < this.width; j++) {
				if (j == 0) {
					if (i == 0) {
						str += "[";
					} else {
						str += " [";
					}
				}
				if (j != this.width - 1) {
					str += this.array[i][j] + ", ";
				} else {
					if (i == this.height - 1) {
						str += this.array[i][j] + "]";
					} else {
						str += this.array[i][j] + "]";
					}
				}
			}
			if (i != this.height - 1) {
				str += "\n";
			}
		}
		str += "])";
		return str;
	}

}
