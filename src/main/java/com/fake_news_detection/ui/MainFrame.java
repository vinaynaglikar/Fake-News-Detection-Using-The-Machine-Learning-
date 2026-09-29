package com.fake_news_detection.ui;

import com.fake_news_detection.model.ModelMetrics;
import com.fake_news_detection.model.NewsPrediction;
import com.fake_news_detection.service.DatasetService;
import com.fake_news_detection.service.ModelService;
import weka.core.Instances;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.io.File;
import java.util.concurrent.ExecutionException;

public class MainFrame extends JFrame {

    private final DatasetService datasetService = new DatasetService();
    private final ModelService modelService = new ModelService();

    private Instances dataset;
    private final JTextArea articleArea = new JTextArea();
    private final JTextArea outputArea = new JTextArea();
    private final JLabel statusLabel = new JLabel("Ready");
    private final JLabel datasetLabel = new JLabel("Dataset: not loaded");
    private final JButton trainButton = new JButton("Train Model");
    private final JButton predictButton = new JButton("Analyze News");
    private final JProgressBar progressBar = new JProgressBar();

    public MainFrame() {
        setTitle("Fake News Detection — Advanced AI Dashboard");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1100, 700);
        setLocationRelativeTo(null);

        buildUi();
    }

    private void buildUi() {
        JPanel root = new JPanel(new BorderLayout(12, 12));
        root.setBorder(new EmptyBorder(15, 15, 15, 15));
        setContentPane(root);

        JPanel header = new JPanel(new BorderLayout());
        JLabel title = new JLabel("FAKE NEWS DETECTION");
        title.setFont(new Font("SansSerif", Font.BOLD, 24));
        JLabel subtitle = new JLabel("Dynamic Weka + Random Forest text classification");
        subtitle.setFont(new Font("SansSerif", Font.PLAIN, 13));

        JPanel heading = new JPanel();
        heading.setLayout(new BoxLayout(heading, BoxLayout.Y_AXIS));
        heading.add(title);
        heading.add(subtitle);
        header.add(heading, BorderLayout.WEST);
        header.add(datasetLabel, BorderLayout.EAST);

        root.add(header, BorderLayout.NORTH);

        articleArea.setLineWrap(true);
        articleArea.setWrapStyleWord(true);
        articleArea.setFont(new Font("SansSerif", Font.PLAIN, 15));
        articleArea.setText("Paste a news article here...");

        outputArea.setEditable(false);
        outputArea.setLineWrap(true);
        outputArea.setWrapStyleWord(true);
        outputArea.setFont(new Font("Monospaced", Font.PLAIN, 13));

        JSplitPane splitPane = new JSplitPane(
                JSplitPane.HORIZONTAL_SPLIT,
                createInputPanel(),
                createOutputPanel()
        );
        splitPane.setResizeWeight(0.55);
        root.add(splitPane, BorderLayout.CENTER);

        JPanel bottom = new JPanel(new BorderLayout(10, 5));
        JPanel actions = new JPanel(new FlowLayout(FlowLayout.LEFT));

        JButton loadButton = new JButton("Load Dataset");
        JButton saveButton = new JButton("Save Model");
        JButton loadModelButton = new JButton("Load Model");
        JButton evaluateButton = new JButton("Evaluate");

        loadButton.addActionListener(e -> loadDataset());
        trainButton.addActionListener(e -> trainModel());
        predictButton.addActionListener(e -> predict());
        saveButton.addActionListener(e -> saveModel());
        loadModelButton.addActionListener(e -> loadModel());
        evaluateButton.addActionListener(e -> evaluate());

        actions.add(loadButton);
        actions.add(trainButton);
        actions.add(predictButton);
        actions.add(evaluateButton);
        actions.add(saveButton);
        actions.add(loadModelButton);

        progressBar.setIndeterminate(false);
        progressBar.setPreferredSize(new Dimension(150, 20));

        bottom.add(actions, BorderLayout.WEST);
        bottom.add(progressBar, BorderLayout.CENTER);
        bottom.add(statusLabel, BorderLayout.EAST);

        root.add(bottom, BorderLayout.SOUTH);
    }

    private JPanel createInputPanel() {
        JPanel panel = new JPanel(new BorderLayout(8, 8));
        panel.setBorder(BorderFactory.createTitledBorder("News Article"));

        JButton clear = new JButton("Clear");
        clear.addActionListener(e -> articleArea.setText(""));

        panel.add(new JScrollPane(articleArea), BorderLayout.CENTER);
        panel.add(clear, BorderLayout.SOUTH);
        return panel;
    }

    private JPanel createOutputPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("AI Analysis"));
        panel.add(new JScrollPane(outputArea), BorderLayout.CENTER);
        return panel;
    }

    private void loadDataset() {
        JFileChooser chooser = new JFileChooser();
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;

        File file = chooser.getSelectedFile();
        runTask("Loading dataset...", () -> {
            dataset = datasetService.load(file);
            return dataset.numInstances() + " records loaded. Class: " + dataset.classAttribute().name();
        }, result -> {
            datasetLabel.setText("Dataset: " + file.getName());
            outputArea.setText(result);
        });
    }

    private void trainModel() {
        if (dataset == null) {
            showError("Load a dataset first.");
            return;
        }

        runTask("Training model...", () -> {
            modelService.train(dataset);
            return "Training completed successfully.";
        }, outputArea::setText);
    }

    private void predict() {
        if (!modelService.isTrained()) {
            showError("Train or load a model first.");
            return;
        }

        String text = articleArea.getText().trim();
        if (text.isBlank()) {
            showError("Enter a news article.");
            return;
        }

        runTask("Analyzing...", () -> modelService.predict(text), prediction -> {
            NewsPrediction p = (NewsPrediction) prediction;
            outputArea.setText(
                    "PREDICTION RESULT\n"
                            + "================\n\n"
                            + "Classification: " + p.label().toUpperCase() + "\n"
                            + "Confidence: " + String.format("%.2f%%", p.confidence() * 100)
                            + "\n\n"
                            + "The result is generated from the trained text classification model."
            );
        });
    }

    private void evaluate() {
        if (dataset == null) {
            showError("Load a dataset first.");
            return;
        }

        runTask("Evaluating model...", () -> modelService.evaluate(dataset), metrics -> {
            ModelMetrics m = (ModelMetrics) metrics;
            outputArea.setText(
                    "MODEL PERFORMANCE\n"
                            + "=================\n\n"
                            + "Accuracy : " + String.format("%.2f%%", m.accuracy() * 100) + "\n"
                            + "Precision: " + String.format("%.4f", m.precision()) + "\n"
                            + "Recall   : " + String.format("%.4f", m.recall()) + "\n"
                            + "F1 Score : " + String.format("%.4f", m.f1Score()) + "\n\n"
                            + m.summary()
            );
        });
    }

    private void saveModel() {
        if (!modelService.isTrained()) {
            showError("Train a model first.");
            return;
        }

        JFileChooser chooser = new JFileChooser();
        chooser.setSelectedFile(new File("fake-news-model.model"));
        if (chooser.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) return;

        File file = chooser.getSelectedFile();
        runTask("Saving model...", () -> {
            modelService.save(file);
            return "Model saved to: " + file.getAbsolutePath();
        }, outputArea::setText);
    }

    private void loadModel() {
        JFileChooser chooser = new JFileChooser();
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;

        File file = chooser.getSelectedFile();
        runTask("Loading model...", () -> {
            modelService.load(file);
            return "Model loaded successfully.";
        }, outputArea::setText);
    }

    private <T> void runTask(
            String message,
            Task<T> task,
            java.util.function.Consumer<T> onSuccess
    ) {
        statusLabel.setText(message);
        progressBar.setIndeterminate(true);
        setButtonsEnabled(false);

        new SwingWorker<T, Void>() {
            @Override
            protected T doInBackground() throws Exception {
                return task.run();
            }

            @Override
            protected void done() {
                progressBar.setIndeterminate(false);
                setButtonsEnabled(true);

                try {
                    onSuccess.accept(get());
                    statusLabel.setText("Ready");
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    showError("Operation interrupted.");
                } catch (ExecutionException e) {
                    showError(e.getCause() == null ? e.getMessage() : e.getCause().getMessage());
                }
            }
        }.execute();
    }

    private void setButtonsEnabled(boolean enabled) {
        trainButton.setEnabled(enabled);
        predictButton.setEnabled(enabled);
    }

    private void showError(String message) {
        statusLabel.setText("Error");
        JOptionPane.showMessageDialog(this, message, "Error", JOptionPane.ERROR_MESSAGE);
    }

    @FunctionalInterface
    private interface Task<T> {
        T run() throws Exception;
    }
}
