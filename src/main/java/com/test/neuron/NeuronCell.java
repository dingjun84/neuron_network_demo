package com.test.neuron;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class NeuronCell {
  private double bias;
  private RealVector weights;
  private Sigmoid sigmoid;

  public NeuronCell(RealVector weights, double bias) {
    this.bias = bias;
    this.weights = weights;
    this.sigmoid = new Sigmoid();
  }

  public void updWeight(RealVector updWeight, double updBias) {
    this.weights = this.weights.subtract(updWeight);
    this.bias -= updBias;
  }

  public RealVector getWeights(){
    return this.weights;
  }

  public double getTotal(RealVector inputs) {
    return inputs.dotProduct(this.weights) + this.bias;
  }

  public double getOutput(double outTotal) {
    return this.sigmoid.value(outTotal);
  }

  public double feedForward(RealVector inputs) {
    double total = inputs.dotProduct(this.weights) + this.bias;
    return this.sigmoid.value(total);
  }

  public static void main(String[] args) {
    RealVector weights = MatrixUtils.createRealVector(new double[] {0, 1});
    RealVector inputs = MatrixUtils.createRealVector(new double[] {2, 3});
    NeuronCell nc = new NeuronCell(weights, 4);
    System.out.println(nc.feedForward(inputs));
  }
}
