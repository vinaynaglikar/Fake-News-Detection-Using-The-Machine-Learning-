package com.fake_news_detection.model;

public record NewsPrediction(String label, double confidence, String text) {
}
