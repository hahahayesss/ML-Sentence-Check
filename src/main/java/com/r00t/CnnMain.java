package com.r00t;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class CnnMain {
    private static final String WORD_VEC_LOCATION = "/Users/hahahayesss/Desktop/sentence_check/GoogleNews-vectors-negative300.bin.gz";
    private static final String DATA_SET_LOCATION = "/Users/hahahayesss/Desktop/sentence_check/";

    private static double learningRate = 0.01;
    private static int seed = 12345;
    private static int batchSize = 32;
    private static int nEpochs = 2;
    private static int vectorSize = 300;
    private static int maxSentenceLength = 256;
    private static int cnnLayerFeatureMaps = 200;

    public static void main(String[] args) throws IOException {
        CnnSentence cnnSentence = new CnnSentence(learningRate, vectorSize, maxSentenceLength, cnnLayerFeatureMaps);

        ComputationGraph network = new ComputationGraph(cnnSentence.createConfig());
        network.init();

        //WRITE
        Arrays.stream(network.getLayers())
                .forEach(l -> System.out.println(l.conf().getLayer().getLayerName() + "\t" + l.numParams()));

        //LOAD WORD VECTORS
        System.out.println("|  Loading word vectors....");
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VEC_LOCATION));

        //CREATE DATASET ITERATOR
        System.out.println("|  Creating dataset iterator....");
        DataSetIterator trainIterator = new MessagesDataSetIterator.Builder(DATA_SET_LOCATION, wordVectors)
                .mode(IteratorMode.TRAINING)
                .batchSize(batchSize)
                .maxSentenceLength(maxSentenceLength)
                .random(new Random(seed))
                .build()
                .getDataSetIterator();
        DataSetIterator testIterator = new MessagesDataSetIterator.Builder(DATA_SET_LOCATION, wordVectors)
                .mode(IteratorMode.TESTING)
                .batchSize(batchSize)
                .maxSentenceLength(maxSentenceLength)
                .random(new Random(seed))
                .build()
                .getDataSetIterator();

        //STARTING TRAINING
        System.out.println("|  Starting UIServer....");
        //UIServer uiServer = UIServer.getInstance();
        //StatsStorage statsStorage = new InMemoryStatsStorage();
        //uiServer.attach(statsStorage);
        //network.setListeners(new StatsListener(statsStorage));
        network.setListeners(new ScoreIterationListener(100), new EvaluativeListener(testIterator, 1, InvocationType.EPOCH_END));

        System.out.println("|  Starting training....");
        network.fit(trainIterator, nEpochs);

        //TEST NETWORK
        testNetwork(testIterator, network);

        //OUTPUT
        ModelSerializer.writeModel(network, ("/Users/hahahayesss/Desktop/sentence_check/" + System.currentTimeMillis() + ".zip"), true);
    }

    public static void testNetwork(DataSetIterator testIter, ComputationGraph network) throws IOException {
        String pathNegative = FilenameUtils.concat(DATA_SET_LOCATION, "test/neg/1_3.txt");
        String content = FileUtils.readFileToString(new File(pathNegative));

        INDArray featuresNegative = ((CnnSentenceDataSetIterator) testIter).loadSingleSentence(content);

        INDArray prediction = network.outputSingle(featuresNegative);
        List<String> labels = testIter.getLabels();

        System.out.println("|  Predictions for test/neg/1_3.txt");
        for (int x = 0; x < labels.size(); x++)
            System.out.println("P(" + labels.get(x) + ") = " + prediction.getDouble(x));
    }
}
