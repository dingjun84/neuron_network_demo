package com.test.neuron;

import java.util.List;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class NeuronLay {
  private List<NeuronCell> layer;

  public NeuronLay(List<NeuronCell> layer) {
    this.layer = layer;
  }

  public RealVector feedForward(RealVector inputs){
    double[] values = new double[layer.size()];
    int i = 0;
    for(NeuronCell nc:layer){
      values[i++] = nc.feedForward(inputs);
    }

    RealVector outputs = MatrixUtils.createRealVector(values);
    return outputs;
  }
  public List<NeuronCell> getLayer(){
    return this.layer;
  }
}
