package org.apache.nlp.sequencelearning.crf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Set;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class CRFDriver{
	private static int converge=0;
	int maxid;
	
	Vector alpha;
	Vector expected;
	double obj;
	int err;
	int zeroone;

	FeatureTemplate featureTemplate;//特征模板
	FeatureExpander featureExpander;//特征扩展器
	FeatureIndexer featureIndexer;//特征索引器
	
	ArrayList<TaggerImpl> taggers;
	CRFLBFGS clbfgs;

	public CRFDriver(){
		this.clbfgs = new CRFLBFGS();
	}
	
	public void initializeCrfDriver(){
		this.maxid = this.featureIndexer.getMaxID();
		
		this.alpha = new DenseVector(this.maxid);// 上次迭代之后，alpha会被更新
		for(int i=0;i<this.alpha.size();i++){
			this.alpha.set(i, 0.0);
		}
		this.expected = new DenseVector(this.maxid);// 全局的expected一致被更新
		for(int i=0;i<this.expected.size();i++){
			this.expected.set(i, 0.0);
		}
		this.obj = 0.0;
		this.err = 0;
		this.zeroone = 0;
	}
	
	/**
	 * 
	 * @param templfile
	 * @param trainfile
	 * @param modelfile
	 * @param textmodelfile
	 * @param xsize
	 * @param maxitr
	 * @param freq
	 * @param eta
	 * @param C
	 * @param algorithm
	 * @return
	 * @throws IOException
	 */
	@SuppressWarnings("resource")
	public CRFModel crf_learn(String templfile, 
						String trainfile, 
						String modelfile,
						boolean textmodelfile, 
						int xsize, int maxitr, double freq, double eta, double C, String algorithm) throws IOException{
		this.featureTemplate = new FeatureTemplate(templfile);//特征模板
		this.featureExpander = new FeatureExpander(this.featureTemplate,xsize);//特征扩展器
		
		//TaggerImpl:crf算法的计算单元，其对应一个句子
		this.taggers= new ArrayList<TaggerImpl>();
		File trainDataPath = new File(trainfile); //
		BufferedReader reader = new BufferedReader(new FileReader(trainDataPath));
		String line = null;
		ArrayList<String> token_list = new ArrayList<String>();
		while ((line = reader.readLine()) != null) {
			TaggerImpl tagger=new TaggerImpl();			
			if( line.trim().equals("") ){
				this.featureExpander.expand(token_list,tagger);//特征扩展
				this.taggers.add(tagger);
				token_list = new ArrayList<String>();
			}else{
				token_list.add(line);
			}
		}
		
		// 特征索引器
		this.featureIndexer = new FeatureIndexer();
		this.featureIndexer.IndexingHStateIndex( this.featureExpander.getHiddenStateSet() );
		for (int i = 0; i < this.taggers.size(); i++) {
			TaggerImpl tagger = this.taggers.get(i);
			this.featureIndexer.IndexingFeatureIndex(tagger);// 索引特征
			this.featureIndexer.Register(tagger);// 注册tagger
		}
		
		this.initializeCrfDriver();
		this.iterateMR(maxitr, eta);
		
		CRFModel model = new CRFModel(featureTemplate,featureExpander,featureIndexer,alpha,expected,obj,err,zeroone);
		return model;
	}

	/**
	 * 
	 * @param templfile
	 * @param testfile
	 * @param model
	 * @param xsize
	 * @return
	 * @throws IOException
	 */
	@SuppressWarnings("resource")
	public boolean crf_test(String templfile, String testfile,
			CRFModel model, int xsize) throws IOException {
		this.featureExpander = model.featureExpander;
		this.featureIndexer = model.featureIndexer;
		Set<String> hsSet = this.featureExpander.getHiddenStateSet();
		String hsArray[] = new String[hsSet.size()];
		int id = 0;
		for( String hiddenState:hsSet){
			hsArray[id] = hiddenState;
			id++;
		}
		
		//TaggerImpl:crf算法的计算单元，其对应一个句子
		File testDataPath = new File( testfile );
		BufferedReader reader = new BufferedReader( new FileReader(testDataPath) );
		String line = null;
		ArrayList<String> token_list = new ArrayList<String>();
		while ( (line = reader.readLine()) != null ) {
//			System.out.println(line);
			TaggerImpl tagger = new TaggerImpl(model.alpha);//为每一个tagger提供其viterbi计算的基础数据:模型的alpha
			if (line.trim().equals("")) {
				this.featureExpander.expand( token_list, tagger );// 特征扩展
				this.featureIndexer.Register( tagger );// 注册tagger	
				tagger.buildLattice();
				tagger.forwardbackward();
				ArrayList<Integer> result = tagger.viterbi();
				
				int tokensNum = token_list.size();
				for(int i=0; i<tokensNum; i++){
					System.out.println( token_list.get(i) + '\t' + hsArray[result.get(i)] );
				}
				
				token_list = new ArrayList<String>();
			} else {
				token_list.add(line);
			}
		}
		
		return true;
	}
	
	/**
	 * @return 
	 * 
	 */
	public double run(){
		double C = 4.0;
		Vector expected_current_iteration = new DenseVector(this.maxid);
		for(int i=0;i<expected_current_iteration.size();i++){
			expected_current_iteration.set(i, 0.0);
		}
		this.obj = 0.0;
		this.err = 0;
		this.zeroone = 0;
		
		// ---遍历所有的tagger
		for(int i=0; i<taggers.size(); i++){
			TaggerImpl taggerImpl = taggers.get(i);
//			System.out.println("taggerImpl.xsize:"+taggerImpl.xsize+"   "+"taggerImpl.ysize:"+taggerImpl.ysize);

			taggerImpl.alpha = this.alpha; //taggerImpl.gradient()中利用this.alpha
			taggerImpl.expected = expected_current_iteration; // update expected_current_iteration
			
			this.obj += taggerImpl.gradient(); // 梯度计算
			
			int error_num = taggerImpl.eval();
			this.err += error_num; 
			if(error_num!=0){
				this.zeroone += 1;
			}
		}
	
//		this.info_after_gradient(this.maxid, this.alpha, this.obj, this.expected);// debug
		for(int i=0;i<this.expected.size();i++){
			this.expected.set(i, 0.0);
			this.expected.set(i, this.expected.get(i)+expected_current_iteration.get(i));
		}
		
		// L-BFGS算法最优化
		int n = this.maxid;//参数个数
		double x [ ]= new double [ n ];//参数向量
		double f ; //目标函数值
		double g [ ] = new double [ n ];//梯度向量
		
		for(int k=0;k<this.alpha.size();k++){//目标函数值和期望向量用罚函数更新
			this.obj += this.alpha.get(k)*this.alpha.get(k)/(2.0*C);
			this.expected.set(k,this.expected.get(k)+this.alpha.get(k)/C);
		}
		/*赋予x,g,f*/
		for(int i=0; i<this.maxid; i++){
			x[i]=this.alpha.get(i);
			g[i]=this.expected.get(i);
		}
		f = this.obj;
		
		this.clbfgs.optimize(n, x, f, g );//x(参数向量)和f(目标函数值)被更新
		
		/**更新alpha和obj*/
		for(int k=0;k<x.length;k++){
			this.alpha.set(k, x[k]);
		}
		this.obj = f;
		
		return this.obj;//
	}
	
	public void iterateMR(int numIterations, double eta) throws IOException{
		System.out.println("Running CRF");
		Double old_obj = new Double(0.0);
		Double obj = new Double(0.0);
		
		int iteration = 1;
		while (iteration <= numIterations) {
			String jobName = "CRF Iterator running iteration " + iteration;
			//System.out.println(jobName);//调试
			
			obj = this.run();
			if (isConverged(iteration, numIterations,eta, old_obj, obj)) {
				break;
			}
			old_obj = obj;
			iteration++;
		}
		System.out.println("迭代结束");
	}
	
	
	/**
	 * 
	 * @param itr
	 * @param maxitr
	 * @param eta
	 * @param old_obj
	 * @param obj
	 * @return
	 * @throws IOException
	 */
	private boolean isConverged(int itr, int maxitr, double eta, double old_obj, double obj) throws IOException {
//		System.out.println("old_obj:"+old_obj+" "+"obj:"+obj);//调试
		double diff = (itr == 1 ? 1.0 : Math.abs(old_obj - obj) / old_obj);// diff是"相对误差限"

		if (diff < eta) {// /eta=9.99e-005
			converge++;// /如果相对误差限diff小于eta，则converge++
		} else {
			converge = 0;
		}
		// 迭代次数itr大于迭代次数限制，或者converge=3；退出
		if (itr > maxitr || converge == 3) {
			return true;
		}
		return false;
	}
	
}
