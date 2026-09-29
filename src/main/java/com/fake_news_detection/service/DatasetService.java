package com.fake_news_detection.service;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;

public class DatasetService {

    public Instances load(File file) throws Exception {
        Instances data = new DataSource(file.getAbsolutePath()).getDataSet();

        if (data.numAttributes() < 2) {
            throw new IllegalArgumentException("Dataset must contain a text attribute and a class attribute.");
        }

        if (data.classIndex() < 0) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        if (!data.classAttribute().isNominal()) {
            throw new IllegalArgumentException("The class attribute must be nominal, for example: fake/real.");
        }

        Attribute textAttribute = findTextAttribute(data);
        if (textAttribute == null) {
            throw new IllegalArgumentException("No string text attribute found in the dataset.");
        }

        return data;
    }

    public Attribute findTextAttribute(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            if (attribute.isString()) {
                return attribute;
            }
        }

        // Fallback: use the first non-class attribute.
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex()) {
                return data.attribute(i);
            }
        }
        return null;
    }
}
