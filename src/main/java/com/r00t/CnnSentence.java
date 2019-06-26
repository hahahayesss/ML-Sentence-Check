package com.r00t;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CnnSentence {
    private double learningRate;
    private int vectorSize;
    private int maxSentenceLength;
    private int cnnLayerFeatureMaps;

    public CnnSentence(double learningRate, int vectorSize, int maxSentenceLength, int cnnLayerFeatureMaps) {
        this.learningRate = learningRate;
        this.vectorSize = vectorSize;
        this.maxSentenceLength = maxSentenceLength;
        this.cnnLayerFeatureMaps = cnnLayerFeatureMaps;
    }

    public ComputationGraphConfiguration createConfig() {
        return new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(0.01))
                .convolutionMode(ConvolutionMode.Same)
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3",
                        new ConvolutionLayer.Builder()
                                .kernelSize(3, vectorSize)
                                .stride(1, vectorSize)
                                .nOut(cnnLayerFeatureMaps)
                                .build(),
                        "input")
                .addLayer("cnn4",
                        new ConvolutionLayer.Builder()
                                .kernelSize(5, vectorSize)
                                .stride(1, vectorSize)
                                .nOut(cnnLayerFeatureMaps)
                                .build(),
                        "input")
                .addLayer("cnn5",
                        new ConvolutionLayer.Builder()
                                .kernelSize(5, vectorSize)
                                .stride(1, vectorSize)
                                .nOut(cnnLayerFeatureMaps)
                                .build(),
                        "input")
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
                .addLayer("globalPool",
                        new GlobalPoolingLayer.Builder()
                                .poolingType(PoolingType.MAX)
                                .dropOut(0.5)
                                .build(),
                        "merge")
                .addLayer("out",
                        new OutputLayer.Builder()
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .nOut(2)
                                .build(),
                        "globalPool")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(maxSentenceLength, vectorSize, 1))
                .build();
    }
}
