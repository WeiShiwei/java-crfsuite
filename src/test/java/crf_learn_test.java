/**
 * Created by weishiwei on 16/5/22.
 */
import java.io.IOException;

import org.apache.nlp.sequencelearning.crf.CRFDriver;
import org.apache.nlp.sequencelearning.crf.CRFModel;

public class crf_learn_test {

    // 指定 templfile,trainfile,testfile,modelfile的文件路径
    static String templfile = "/Users/weishiwei/IdeaProjects/java-crfsuite/src/test/files/template";
    static String trainfile = "/Users/weishiwei/IdeaProjects/java-crfsuite/src/test/files/train.data";
    static String testfile = "/Users/weishiwei/IdeaProjects/java-crfsuite/src/test/files/test.data";
    static String modelfile = "/Users/weishiwei/IdeaProjects/java-crfsuite/src/test/files/model.data";

    static CRFModel model;
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        crf_learn();
		crf_test();
    }

    public static boolean crf_learn() throws IOException{
        System.out.println("Running CRF learn...");
        CRFDriver driver = new CRFDriver();

        boolean textmodelfile = false;
        int xsize = 3;
        int maxitr = 10000;
        double freq = 1.0;
        double eta = 0.0001;
        double C = 1.0;
        String algorithm = "L-BFGS";
        // crf_learn
        model = driver.crf_learn(templfile, trainfile, modelfile, textmodelfile, xsize, maxitr, freq, eta, C, algorithm);

        // 存储模型
//        model.dump(modelfile); //花费较多的时间

        return true;
    }

    /**
     *
     * @return
     * @throws IOException
     */
    public static boolean crf_test() throws IOException{
        System.out.println("\nRunning CRF test...");
        CRFDriver driver = new CRFDriver();

        int xsize = 3;
        driver.crf_test(templfile, testfile, load_model(modelfile), xsize);

        return true;
    }

    public static CRFModel load_model(String modelfile){
        return model;
    }

}
