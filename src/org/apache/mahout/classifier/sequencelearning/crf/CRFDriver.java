package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.IOException;
import java.util.ArrayList;

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

	ArrayList<TaggerImpl> taggers = new ArrayList<TaggerImpl>();
	CRFLBFGS clbfgs;
	
	/**
	 * 
	 * @param taggers
	 * @param maxid
	 */
	public CRFDriver(ArrayList<TaggerImpl> taggers, int maxid){
		this.maxid = maxid;
		
		this.alpha = new DenseVector(this.maxid);
		for(int i=0;i<this.alpha.size();i++){
			this.alpha.set(i, 0.0);
		}
		this.expected = new DenseVector(this.maxid);
		for(int i=0;i<this.expected.size();i++){
			this.expected.set(i, 0.0);
		}
//		this.obj = new DenseVector(1);
//		this.err = new DenseVector(1);
//		this.zeroone = new DenseVector(1);
		
//		this.alpha = new double[this.maxid]; // 上次迭代之后，alpha会被更新
//		this.expected = new double[this.maxid]; // 上次迭代之后，expected会被初始化为元素为0的数组
		this.obj = 0.0;
		this.err = 0;
		this.zeroone = 0;
		
		this.taggers= taggers;
		this.clbfgs = new CRFLBFGS();
	}

	public void info_after_gradient(int n, Vector alpha2, double f, Vector expected2) {

		System.out.println("n:" + n);
		System.out.println("x:");
		for (int i = 0; i < alpha2.size(); i++) {
//			System.out.println("x[" + i + "]=" + alpha2[i]);
			System.out.println("x[" + i + "]=" + alpha2.get(i));
		}

		System.out.println("f:" + f);

		System.out.println("g:");
		for (int i = 0; i < expected2.size(); i++) {
			System.out.println("g[" + i + "]=" + expected2.get(i));
		}
		System.out.println();
	}

	public void info_before_lbfgs(int n,double x[],double f,double g[]){
		
		System.out.println("n:"+n);
		System.out.println("x:");
		for(int i=0;i<x.length;i++){
			System.out.println("x["+i+"]="+x[i]);
		}

		System.out.println("f:"+f);
		
		System.out.println("g:");
		for(int i=0;i<g.length;i++){
			System.out.println("g["+i+"]="+g[i]);
		}
		System.out.println();
	}
	/**
	 * @return 
	 * 
	 */
	public double run(){
		double C = 4.0;
		// ----------------------------------------------------------------------------------------模拟crfpp中的线程0
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
		// ----------------------------------------------------------------------------------------模拟crfpp中的runCRF
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
		Double old_obj = new Double(0.0);
		Double obj = new Double(0.0);
		
		int iteration = 1;
		while (iteration <= numIterations) {
			String jobName = "CRF Iterator running iteration " + iteration;
//			System.out.println(jobName);//调试
			
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
			return true; // 3 is ad-hoc[ad-hoc:特定的、自定义?]
		}
		return false;
	}
	
}
