package com.fake_news_detection.model;

public record ModelMetrics(
        double accuracy,
        double precision,
        double recall,
        double f1Score,
        String summary
) {
}
