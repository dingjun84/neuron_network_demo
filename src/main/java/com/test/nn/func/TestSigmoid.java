package com.test.nn.func;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class TestSigmoid {
  public static void main(String[] args){
    /*Sigmoid sigmoid = new Sigmoid();
    System.out.println(sigmoid.value(100));
    System.out.println(sigmoid.value(-100));
    System.out.println(sigmoid.value(-1));
    System.out.println(sigmoid.value(1));*/
    RealMatrix weights = MatrixUtils.createRealMatrix(new double[][]{{0,1}});
    RealMatrix data = MatrixUtils.createRealMatrix(new double[][]{{2,3}});
    double bias = 4;
    System.out.println(feedforward(data,weights,bias));
  }

  public static double feedforward(RealMatrix data,RealMatrix weights,double bias){
    double total = data.getRowVector(0).dotProduct(weights.getRowVector(0))+bias;
    Sigmoid sigmoid = new Sigmoid();
    return sigmoid.value(total);
  }
}
