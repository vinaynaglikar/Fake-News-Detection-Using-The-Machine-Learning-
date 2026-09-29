# Advanced Fake News Detection

A dynamic Java desktop application using Weka, StringToWordVector and Random Forest.

## Features

- Dynamic ARFF dataset loading
- Automatic text attribute detection
- Dynamic class attribute detection
- Random Forest text classification
- TF-IDF text features
- Stop-word filtering
- Model training
- Cross-validation evaluation
- Accuracy, precision, recall and F1 score
- Single article prediction
- Model save/load
- Non-blocking Swing UI with background workers
- No hard-coded fake/real labels

## Dataset format

Use an ARFF file with:

- one string attribute containing the news article
- one nominal class attribute as the final attribute

Example:

@attribute text string
@attribute class {fake,real}

The application automatically detects the text attribute and class attribute.

## Run

Install Java 17+ and Maven.

```bash
mvn clean compile
mvn exec:java
```

## Important

This project is a machine-learning classifier. Its accuracy depends heavily on the quality, size and representativeness of the training dataset.
