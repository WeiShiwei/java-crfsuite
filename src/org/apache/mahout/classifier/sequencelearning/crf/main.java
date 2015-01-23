package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.IOException;

public class main {

	static CRFModel model;
	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		crf_learn();
//		crf_test();
	}
	
	public static boolean crf_learn() throws IOException{
		System.out.println("Running CRF learn...");
		CRFDriver driver = new CRFDriver();
		
		String templfile = "./template";
		String trainfile = "./train.data";
		String modelfile = "./model.data";
		boolean textmodelfile = false;
		int xsize = 3;
		int maxitr = 10000;
		double freq = 1.0;
		double eta = 0.0001;
		double C = 1.0;
		String algorithm = "L-BFGS";
		model = driver.crf_learn(templfile, trainfile, modelfile, textmodelfile, xsize, maxitr, freq, eta, C, algorithm);
		
		model.dump(modelfile);
		return true;
	}
	
	public static boolean crf_test() throws IOException{
		System.out.println("\nRunning CRF test...");
		CRFDriver driver = new CRFDriver();
		
		String templfile = "./template";
		String testfile = "./test.data";
		String modelfile = "./model.data";
		int xsize = 3;
		
		driver.crf_test(templfile, testfile, load_model(modelfile), xsize);
		
		return true;
	}
	
	public static CRFModel load_model(String modelfile){
		return model;
	}

}
