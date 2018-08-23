package nn;

/*************************************************************************
 * 
 * @author Yuyan Wang
 * @date 2018.8
 * 
 * Flags of some variables.
 * These flags are used for forward propagation or back propagation.
 * These flags also represent default values.
 * 
 *************************************************************************/

public class Const {
	
	// Initialize flags and some default value.
	public static boolean random = false;
	public static double weightRate = 0.01;
	public static boolean He = false;
	public static boolean Xavier = false;
	
	// Activation flags.
	public static boolean tanh = false;
}
