package com.r00t;

import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.*;

public class MessagesDataSetIterator {
    private DataSetIterator dataSetIterator;

    private MessagesDataSetIterator(String dataSetPath, IteratorMode mode, WordVectors wordVectors,
                                    int batchSize, int maxSentenceLength, Random random) {
        Map<String, List<File>> messageFiles = new HashMap<>();
        messageFiles.put("Positive", getFiles(dataSetPath + mode.getPath() + "/pos"));
        messageFiles.put("Negative", getFiles(dataSetPath + mode.getPath() + "/neg"));

        LabeledSentenceProvider provider = new FileLabeledSentenceProvider(messageFiles, random);
        dataSetIterator = new CnnSentenceDataSetIterator.Builder()
                .sentenceProvider(provider)
                .wordVectors(wordVectors)
                .minibatchSize(batchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .build();
    }

    private List<File> getFiles(String filePath) {
        return Arrays.asList(Objects.requireNonNull(new File(filePath).listFiles()));
    }

    public DataSetIterator getDataSetIterator() {
        return dataSetIterator;
    }

    public static class Builder {
        private IteratorMode mode;
        private String dataSetPath;
        private WordVectors wordVectors;
        private int batchSize;
        private int maxSentenceLength;
        private Random random;

        public Builder(String dataSetPath, WordVectors wordVectors) {
            this.dataSetPath = dataSetPath;
            this.wordVectors = wordVectors;

            this.mode = IteratorMode.TRAINING;
            this.batchSize = 32;
            this.maxSentenceLength = 256;
            this.random = new Random(12345);
        }

        public Builder mode(IteratorMode mode) {
            this.mode = mode;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder maxSentenceLength(int maxSentenceLength) {
            this.maxSentenceLength = maxSentenceLength;
            return this;
        }

        public Builder random(Random random) {
            this.random = random;
            return this;
        }

        public MessagesDataSetIterator build() {
            return new MessagesDataSetIterator(
                    dataSetPath, mode, wordVectors, batchSize, maxSentenceLength, random
            );
        }
    }
}
