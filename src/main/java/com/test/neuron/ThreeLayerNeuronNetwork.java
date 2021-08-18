package com.test.neuron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Max;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class ThreeLayerNeuronNetwork {
  private int inputSize;
  private int hiddenLayerSize;
  private int outputSize;

  private NeuronLay hiddenLay;
  private NeuronLay outLay;
  private Sigmoid sigmoid = new Sigmoid();

  public ThreeLayerNeuronNetwork(int inputSize, int hiddenLayerSize, int outputSize) {
    this.inputSize = inputSize;
    this.hiddenLayerSize = hiddenLayerSize;
    this.outputSize = outputSize;
  }

  public void init(RealVector weights,double bias){
    if(inputSize != weights.getDimension()){
      throw new RuntimeException("inputSize not equals weights dimension");
    }

    if(hiddenLayerSize != weights.getDimension()){
      throw new RuntimeException("hiddenLaySize not equals weights dimension");
    }
    List<NeuronCell> hl = new ArrayList<>();
    for(int i=0;i<hiddenLayerSize;i++){
      NeuronCell nc1 = new NeuronCell(weights,bias);
      hl.add(nc1);
    }

    List<NeuronCell> ol = new ArrayList<>();
    for(int i=0;i<outputSize;i++){
      NeuronCell nc1 = new NeuronCell(weights,bias);
      ol.add(nc1);
    }
    this.hiddenLay = new NeuronLay(hl);
    this.outLay = new NeuronLay(ol);
  }

  public double feedForward(RealVector inputs){
    RealVector outputs = this.hiddenLay.feedForward(inputs);
    outputs = this.outLay.feedForward(outputs);
    return outputs.getEntry(0);
  }

  public double mseLoss(RealVector trueVector,RealVector preVector){
    RealVector diffVector = trueVector.subtract(preVector);
    double sum = 0;
    for(int i=0;i<diffVector.getDimension();i++){
      sum += diffVector.getEntry(i)*diffVector.getEntry(i);
    }
    return sum/diffVector.getDimension();
  }

  public double derivSigmoid(double x){
    double fx = sigmoid.value(x);
    return fx*(1-fx);
  }

  public void trainOnce(RealMatrix data,double learnRate){
    int rows = data.getRowDimension();
    for(int i=0;i<rows;i++){
      trainOneVector(data.getRowVector(i),learnRate);
    }
  }

  public void trainOneVector(RealVector vector,double learnRate){
    List<NeuronCell> hList = this.hiddenLay.getLayer();

    NeuronCell h1 = hList.get(0);
    NeuronCell h2 = hList.get(1);

    NeuronCell out1 = this.outLay.getLayer().get(0);

    double h1Total = h1.getTotal(MatrixUtils.createRealVector(new double[]{vector.getEntry(0),vector.getEntry(1)}));
    double h1Output = h1.getOutput(h1Total);

    double h2Total = h2.getTotal(MatrixUtils.createRealVector(new double[]{vector.getEntry(0),vector.getEntry(1)}));
    double h2Output = h2.getOutput(h2Total);

    double o1Total = out1.getTotal(MatrixUtils.createRealVector(new double[]{h1Output,h2Output}));
    double o1Output = out1.getOutput(o1Total);

    double predResult = o1Output;
    //损失函数的导数
    double dResultPred = -2*(vector.getEntry(2) - predResult);

    //输出层偏导
    double dOutLayCell1W1 = h1Output*derivSigmoid(o1Total);
    double dOutLayCell1W2 = h2Output*derivSigmoid(o1Total);
    double dOutLayCell1W3 = derivSigmoid(o1Total);

    //隐藏层与输出层传递值的偏导
    double dHiddenOut1 = out1.getWeights().getEntry(0) * derivSigmoid(o1Total);
    double dHiddenOut2 = out1.getWeights().getEntry(1) * derivSigmoid(o1Total);

    //隐藏层参数变量的偏导
    double dHiddenLayCell1W1 = vector.getEntry(0)*derivSigmoid(h1Total);
    double dHiddenLayCell1W2 = vector.getEntry(1)*derivSigmoid(h1Total);
    double dHiddenLayCell1W3 = derivSigmoid(h1Total);

    double dHiddenLayCell2W1 = vector.getEntry(0)*derivSigmoid(h2Total);
    double dHiddenLayCell2W2 = vector.getEntry(1)*derivSigmoid(h2Total);
    double dHiddenLayCell2W3 = derivSigmoid(h2Total);


    double updH1W1 = learnRate*dResultPred*dHiddenOut1*dHiddenLayCell1W1;
    double updH1W2 = learnRate*dResultPred*dHiddenOut1*dHiddenLayCell1W2;
    double updH1Bias = learnRate*dResultPred*dHiddenOut1*dHiddenLayCell1W3;

    double updH2W1 = learnRate*dResultPred*dHiddenOut2*dHiddenLayCell2W1;
    double updH2W2 = learnRate*dResultPred*dHiddenOut2*dHiddenLayCell2W2;
    double updH2Bias = learnRate*dResultPred*dHiddenOut2*dHiddenLayCell2W3;

    double updO1W1 = learnRate*dResultPred*dOutLayCell1W1;
    double updO1W2 = learnRate*dResultPred*dOutLayCell1W2;
    double updO1Bias = learnRate*dResultPred*dOutLayCell1W3;

    h1.updWeight(MatrixUtils.createRealVector(new double[]{updH1W1,updH1W2}),updH1Bias);
    h2.updWeight(MatrixUtils.createRealVector(new double[]{updH2W1,updH2W2}),updH2Bias);
    out1.updWeight(MatrixUtils.createRealVector(new double[]{updO1W1,updO1W2}),updO1Bias);

  }


  public static void main(String[] args){
    ThreeLayerNeuronNetwork nn = new ThreeLayerNeuronNetwork(2,2,1);
    nn.init(MatrixUtils.createRealVector(new double[]{0,1}),0);
    double v = nn.feedForward(MatrixUtils.createRealVector(new double[]{2,3}));
    System.out.println(v);

    RealVector trueVector = MatrixUtils.createRealVector(new double[]{1,0,0,1});
    RealVector preVector = MatrixUtils.createRealVector(new double[]{0,0,0,0});
    System.out.println(nn.mseLoss(trueVector,preVector));


    double[][] data = new double[][]{
        {-1,-1,1},{25,6,0},{17,4,0},{-15,-6,1}
    };
    double[] predResults = new double[data.length];
    for(int k=0;k<1000;k++){
      nn.trainOnce(MatrixUtils.createRealMatrix(data),0.1);
      if(k%10 == 0){
        int i = 0;
        for(double[] row : data){
          predResults[i++] = nn.feedForward(MatrixUtils.createRealVector(new double[]{row[0],row[1]}));
        }

        double mseLossV = nn.mseLoss(MatrixUtils.createRealMatrix(data).getColumnVector(2),MatrixUtils.createRealVector(predResults));
        //对比损失函数
        System.out.println("mseLossV:"+mseLossV);
      }
    }

    System.out.println(Arrays.toString(predResults));
  }
}
