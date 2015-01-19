package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class main {

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		// TODO Auto-generated method stub
//		BuildTaggerImplMapper
		int xsize = 3;
		
		String templatePath = "./template";
		FeatureTemplate featureTemplate = new FeatureTemplate(templatePath);
		
		FeatureExpander featureExpander = new FeatureExpander(featureTemplate,xsize);
		ArrayList<TaggerImpl> taggers= new ArrayList<TaggerImpl>();
		
		File file = new File("./train.data");
		BufferedReader reader = new BufferedReader(new FileReader(file));
		String line = null;
		ArrayList<String> token_list = new ArrayList<String>();
		while ((line = reader.readLine()) != null) {
			TaggerImpl tagger=new TaggerImpl();			
			if( line.trim().equals("") ){
				featureExpander.Expand(token_list,tagger);
				taggers.add(tagger);//				
				token_list = new ArrayList<String>();
			}else{
				token_list.add(line);
			}
		}
//		---------------------------------------
		String hiddenStateList = "B-ADJP@@B-NP@@B-PP@@B-SBAR@@B-VP@@I-NP@@I-VP@@O";
//		String hiddenStateList = "B-NP@@B-PP@@I-NP@@B-VP@@I-VP@@B-SBAR@@O@@B-ADJP";
//		String hiddenStateList = "B-ADJP@@B-NP@@B-PP@@B-SBAR@@B-VP@@I-NP@@I-VP@@O";
//		String hiddenStateList = "B-ADJP@@B-ADVP@@B-NP@@B-PP@@B-PRT@@B-SBAR@@B-VP@@I-ADJP@@I-ADVP@@I-NP@@I-PP@@I-SBAR@@I-VP@@O";
		String[] hidden_state_list = hiddenStateList.split("@@");
		FeatureIndex featureIndex=new FeatureIndex();
		featureIndex.IndexingHStateIndex(hidden_state_list);
		
		for(int i=0;i<taggers.size();i++){
			TaggerImpl tagger = taggers.get(i);
			featureIndex.IndexingFeatureIndex(tagger);
			featureIndex.Register(tagger);
		}
		
		int maxid=featureIndex.getMaxID();
		CRFModel model=new CRFModel(maxid,"featureIndexURI");
//		---------------------------------------
		System.out.println("Running CRF");
		int maxitr = 10000;
		double eta = 0.0001;
		double C = 1.0;
//		int shrinking_size = 20;
//		String algorithm = "L-BFGS";
		CRFDriver driver = new CRFDriver(taggers,maxid);
		driver.iterateMR(maxitr, eta);
		
		

	}

}
