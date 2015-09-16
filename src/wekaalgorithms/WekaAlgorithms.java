/*
 * The MIT License
 *
 * Copyright 2015 brunoocasali.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package wekaalgorithms;

import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author brunoocasali
 */
public class WekaAlgorithms {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
//    public static void main(String[] args)  {
        //run();
       
//    }

    private static void run() throws Exception{
        DataSource source = new DataSource("src/files/powerpuffgirls.arff");

        int folds = 10;
        int runs = 30;

        HashMap<String, Classifier> hash = new HashMap<>();

        hash.put("J48", new J48());
        hash.put("NaiveBayes", new NaiveBayes());
        hash.put("IBk=1", new IBk(1));
        hash.put("IBk=3", new IBk(3));
        hash.put("MultilayerPerceptron", new MultilayerPerceptron());
        
//        LibSVM svm = new LibSVM();
//        svm.setOptions(new String[]{"-S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 0.40 -C 1.0 -E 0.001 -P 0.1"});
        
//        hash.put("LibSVM", svm);

        Instances data = source.getDataSet();
        data.setClassIndex(4);

        System.out.println("#seed \t correctly instances \t percentage of corrects\n");

        for (Entry<String, Classifier> entry : hash.entrySet()) {
            System.out.println("\n Algorithm: " + entry.getKey() + "\n");

            for (int i = 1; i <= runs; i++) {
                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(entry.getValue(), data, folds, new Random(i));

                System.out.println(summary(eval));
            }
        }
    }
            
    private static String summary(Evaluation eval) {
        return Utils.doubleToString(eval.pctCorrect(), 12, 4);
    }
}
