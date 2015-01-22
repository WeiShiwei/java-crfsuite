package org.apache.mahout.classifier.sequencelearning.crf;

import org.apache.mahout.math.Vector;

public class CRFModel {
	final int version = 100;
	final double const_factor=1.0;
	
	/*特征模板*/
	FeatureTemplate featureTemplate;
	/*特征扩展器*/
	FeatureExpander featureExpander;
	/*特征索引器*/
	FeatureIndexer featureIndexer;
	
	/*语料库中提取的特征的最大数量，alpha和expected的维度都是maxid*/
	int maxid;
	/*标注语料库横向的维度,初始化后不能改变*/
	int xsize;
	
	/*特征的权重系数*/
	Vector alpha;
	/*特征的期望（模型期望与经验期望）*/
	Vector expected;
	/*目标函数值*/
	double obj;
	/*err统计的是当前迭代下，该线程总共预测token错误的个数*/
	int err;
	/*zeroone统计的是当前迭代下，该线程总共预测sentence错误的个数*/
	int zeroone;
	
	
	public CRFModel(FeatureTemplate featureTemplate,
					FeatureExpander featureExpander,
					FeatureIndexer featureIndexer,
					Vector alpha,
					Vector expected,
					double obj,
					int err,
					int zeroone){
		this.featureTemplate = featureTemplate;
		this.featureExpander = featureExpander;
		this.featureIndexer = featureIndexer;
		this.alpha = alpha;
		this.expected = expected;
		this.obj = obj;
		this.err = err;
		this.zeroone = zeroone;
	}
	
}