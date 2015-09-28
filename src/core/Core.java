package core;

import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_highgui.cvLoadImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author bruno
 */
public class Core {

    File image = null;

    private int CUT_NOTE = 75;
    
    public Core(File image) {
        this.image = image;
    }

    public String run() throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/files/powerpuffgirls.arff");

        HashMap<String, Classifier> hash = new HashMap<>();

        hash.put("J48", new J48());
        hash.put("NaiveBayes", new NaiveBayes());
        hash.put("IBk=1", new IBk(1));
        hash.put("IBk=3", new IBk(3));
        hash.put("MultilayerPerceptron", new MultilayerPerceptron());

        LibSVM svm = new LibSVM();
        hash.put("LibSVM", svm);
        Instances ins = source.getDataSet();

        ins.setClassIndex(4);

        StringBuilder sb = new StringBuilder();

        int blossom = 0;
        int bubbles = 0;
        
        Instance test = null;
        
        for (Map.Entry<String, Classifier> entry : hash.entrySet()) {
            Classifier c = entry.getValue();
            c.buildClassifier(ins);

            test = new Instance(5);

            float[] array = classifyImage();

            test.setDataset(ins);
            test.setValue(0, array[0]);
            test.setValue(1, array[1]);
            test.setValue(2, array[2]);
            test.setValue(3, array[3]);

            double prob[] = c.distributionForInstance(test);

            sb.append("<em>");
            sb.append(entry.getKey());
            sb.append(":</em>");
            sb.append("<br/>");

            for (int i = 0; i < prob.length; i++) {
                String value = test.classAttribute().value(i);

                if (getRoundedValue(prob[i]) >= CUT_NOTE){
                    if (getClassValue(value))
                        blossom++;
                    else
                        bubbles++;
                }

                sb.append(getClassName(value));
                sb.append(": ");
                sb.append("<strong>");
                sb.append(getRoundedValue(prob[i]) < CUT_NOTE ? "Rejeitado!" : getValueFormatted(prob[i]));
                sb.append("</strong>");
                sb.append("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;");
            }

            sb.append("<br/>");
            
            System.out.println("blossom: " + blossom);
            System.out.println("bubbles: " + bubbles);
            System.out.println("=================\n");
        }
        
        sb.append(blossom > bubbles ? "<h3>é a Florzinha!</h3>" : "<h3>é a Lindinha!</h3>");
        
        blossom = 0;
        bubbles = 0;
        
        return sb.toString();
    }

    private float[] classifyImage() {
        int red, green, blue, altura;
        float bluePowerBubbles = 0;
        float yellowHairBubbles = 0;
        float redPowerBlossom = 0;
        float orangeHairBlossom = 0;

        IplImage imagemOriginal = new IplImage();
        imagemOriginal = cvLoadImage(this.image.getAbsolutePath());
        ByteBuffer buf = imagemOriginal.getByteBuffer();

        for (altura = 0; altura < imagemOriginal.height(); altura++) {
            for (int largura = 0; largura < imagemOriginal.width(); largura++) {

                blue = buf.get(altura * imagemOriginal.width() * 3 + largura) & 0xFF;
                green = buf.get(altura * imagemOriginal.width() * 3 + largura + 1) & 0xFF;
                red = buf.get(altura * imagemOriginal.width() * 3 + largura + 2) & 0xFF;

                // blue power Bubbles!
                if (blue >= 213 && blue <= 243 && green >= 149 && green <= 179 && red >= 41 && red <= 71) {
                    bluePowerBubbles++;
                }

                // Bubbles blonde hair!
                if (altura < (imagemOriginal.height() / 2)) {
                    if (blue >= 32 && blue <= 62 && green >= 232 && green <= 262 && red >= 238 && red <= 268) {
                        yellowHairBubbles++;
                    }
                }

                // red power Blossom!
                if (blue >= 10 && blue <= 65 && green >= 0 && green <= 47 && red >= 220 && red <= 255) {
                    redPowerBlossom++;
                }

                // Blossom "red head" hair!
                if (altura < (imagemOriginal.height() / 2)) {
                    if (blue >= 0 && blue <= 42 && green >= 60 && green <= 98 && red >= 220 && red <= 255) {
                        orangeHairBlossom++;
                    }
                }
            }
        }

        return new float[]{bluePowerBubbles, yellowHairBubbles, redPowerBlossom, orangeHairBlossom};
    }

    private String getValueFormatted(double value) {
        return Math.round(value * 100) + "%";
    }

    private long getRoundedValue(double value) {
        return Math.round(value * 100);
    }

    private String getClassName(String value) {
        return value.equals("1.0") ? "Florzinha" : "Lindinha";
    }

    private boolean getClassValue(String value) {
        return value.equals("1.0");
    }
}
