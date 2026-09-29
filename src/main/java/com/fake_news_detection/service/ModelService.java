package com.fake_news_detection.service;

import com.fake_news_detection.model.ModelMetrics;
import com.fake_news_detection.model.NewsPrediction;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.util.Random;

public class ModelService {

    private FilteredClassifier model;
    private Instances structure;
    private Attribute textAttribute;

    public void train(Instances data) throws Exception {
        textAttribute = findTextAttribute(data);

        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true);
        filter.setWordsToKeep(5000);
        filter.setMinTermFreq(1);
        filter.setIDFTransform(true);
        filter.setTFTransform(true);
        filter.setStopwordsHandler(new weka.core.stopwords.Rainbow());

        RandomForest forest = new RandomForest();
        forest.setNumIterations(200);
        forest.setNumExecutionSlots(Math.max(1, Runtime.getRuntime().availableProcessors() - 1));

        model = new FilteredClassifier();
        model.setFilter(filter);
        model.setClassifier(forest);
        model.buildClassifier(data);

        structure = new Instances(data, 0);
    }

    public ModelMetrics evaluate(Instances data) throws Exception {
        if (data.numInstances() < 2) {
            throw new IllegalArgumentException("At least two records are required for evaluation.");
        }

        Evaluation evaluation = new Evaluation(data);
        int folds = Math.min(10, data.numInstances());
        evaluation.crossValidateModel(buildFreshModel(data), data, folds, new Random(42));

        return new ModelMetrics(
                evaluation.pctCorrect() / 100.0,
                evaluation.weightedPrecision(),
                evaluation.weightedRecall(),
                evaluation.weightedFMeasure(),
                evaluation.toSummaryString()
        );
    }

    private FilteredClassifier buildFreshModel(Instances data) throws Exception {
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true);
        filter.setWordsToKeep(5000);
        filter.setMinTermFreq(1);
        filter.setIDFTransform(true);
        filter.setTFTransform(true);
        filter.setStopwordsHandler(new weka.core.stopwords.Rainbow());

        RandomForest forest = new RandomForest();
        forest.setNumIterations(200);

        FilteredClassifier fresh = new FilteredClassifier();
        fresh.setFilter(filter);
        fresh.setClassifier(forest);
        fresh.buildClassifier(data);
        return fresh;
    }

    public NewsPrediction predict(String text) throws Exception {
        ensureTrained();

        Instance instance = new DenseInstance(structure.numAttributes());
        instance.setDataset(structure);
        instance.setValue(textAttribute, text);
        instance.setMissing(structure.classIndex());

        double index = model.classifyInstance(instance);
        double[] distribution = model.distributionForInstance(instance);
        double confidence = distribution.length > (int) index ? distribution[(int) index] : 0.0;
        String label = structure.classAttribute().value((int) index);

        return new NewsPrediction(label, confidence, text);
    }

    public void save(File file) throws Exception {
        ensureTrained();
        weka.core.SerializationHelper.write(file.getAbsolutePath(), model);
        weka.core.SerializationHelper.write(
                file.getAbsolutePath() + ".structure",
                structure
        );
    }

    public void load(File file) throws Exception {
        model = (FilteredClassifier) weka.core.SerializationHelper.read(file.getAbsolutePath());
        structure = (Instances) weka.core.SerializationHelper.read(file.getAbsolutePath() + ".structure");
        textAttribute = findTextAttribute(structure);
    }

    public boolean isTrained() {
        return model != null && structure != null;
    }

    private void ensureTrained() {
        if (!isTrained()) {
            throw new IllegalStateException("Train or load a model first.");
        }
    }

    private Attribute findTextAttribute(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isString()) {
                return data.attribute(i);
            }
        }
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex()) {
                return data.attribute(i);
            }
        }
        throw new IllegalArgumentException("Text attribute not found.");
    }
}
