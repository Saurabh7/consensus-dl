package org.deeplearning4j.examples.unsupervised.anomalydetection;

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class VisualizerUtil {
    private double imageScale;
    private List<INDArray> digits;  //Digits (as row vectors), one per INDArray
    private String title;
    private int gridWidth;

    public VisualizerUtil(double imageScale, List<INDArray> digits, String title ) {
        this(imageScale, digits, title, 5);
    }

    public VisualizerUtil(double imageScale, List<INDArray> digits, String title, int gridWidth ) {
        this.imageScale = imageScale;
        this.digits = digits;
        this.title = title;
        this.gridWidth = gridWidth;
    }

    public void visualize(){
        JFrame frame = new JFrame();
        frame.setTitle(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(0,gridWidth));

        List<JLabel> list = getComponents();
        for(JLabel image : list){
            panel.add(image);
        }

        frame.add(panel);
        frame.setVisible(true);
        frame.pack();
    }

    private List<JLabel> getComponents(){
        List<JLabel> images = new ArrayList<>();
        for( INDArray arr : digits ){
            BufferedImage bi = new BufferedImage(28,28,BufferedImage.TYPE_BYTE_GRAY);
            for( int i=0; i<784; i++ ){
                bi.getRaster().setSample(i % 28, i / 28, 0, (int)(255*arr.getDouble(i)));
            }
            ImageIcon orig = new ImageIcon(bi);
            Image imageScaled = orig.getImage().getScaledInstance((int)(imageScale*28),(int)(imageScale*28),Image.SCALE_REPLICATE);
            ImageIcon scaled = new ImageIcon(imageScaled);
            images.add(new JLabel(scaled));
        }
        return images;
    }
}
