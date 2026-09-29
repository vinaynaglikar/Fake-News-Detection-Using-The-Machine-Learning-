package com.fake_news_detection;

import javax.swing.SwingUtilities;
import com.fake_news_detection.ui.MainFrame;

public class App {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new MainFrame().setVisible(true));
    }
}
