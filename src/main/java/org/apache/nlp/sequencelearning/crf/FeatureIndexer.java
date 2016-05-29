package org.apache.nlp.sequencelearning.crf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public class FeatureIndexer {

	class Pair{
		int ID=0;
		int Freq=0;
		public Pair(int id,int freq){
			ID=id;
			Freq=freq;
		}
	}
	private int ysize=0;//隐藏状态集合的大小，在IndexingHiddenState函数中更新设置
	private int maxid=0;//特征函数的个数，在IndexingFeatureIndex函数中更新设置
	//注意：如果Writable类型作为Map容器的类型会出错的，遍历的时候全部是最后一个元素
	private Map<String, Pair> FeatureIndexMap = new HashMap<String, Pair>();
	private Map<String, Integer> HStateIndexMap = new HashMap<String, Integer>();
	
	public void IndexingHStateIndex(Set<String> hiddenStateTreeSet){
		ysize = hiddenStateTreeSet.size();
		int i=0;
		for(String hidden_state : hiddenStateTreeSet){
			if(hidden_state!=""){
				HStateIndexMap.put(hidden_state, i);
			}
			i++;
		}
	}
	public void IndexingFeatureIndex(TaggerImpl tagger){
		for(ArrayList<String> featureList : tagger.xStr){
			for(String feature : featureList){
				if(!FeatureIndexMap.containsKey(feature)){
					Pair idFreq=new Pair(maxid,1);
					FeatureIndexMap.put(feature,idFreq);
					if(feature.startsWith("U")){
						maxid+=ysize;
					}else{
						maxid+=ysize*ysize;
					}
				}else{
					FeatureIndexMap.get(feature).Freq++;
				}
			}
		}
		
	}
	
	public void Register(TaggerImpl tagger){
		for(ArrayList<String> featurelist : tagger.xStr){
			ArrayList<Integer> fvector=new ArrayList<Integer>();
			for(String feature:featurelist){
				if( FeatureIndexMap.containsKey(feature) ){
					fvector.add(FeatureIndexMap.get(feature).ID);
				}
			}
			tagger.x.add(fvector);
		}
		
		for(String hiddenstate : tagger.answerStr){
			tagger.answer.add(HStateIndexMap.get(hiddenstate));
		}
		tagger.xsize=tagger.answerStr.size();
		tagger.ysize=ysize;
	}
	
	public int getMaxID(){
		return maxid;
	}
	
	
	public void infoFeatureIndexMap(){
		Iterator<String> iter = FeatureIndexMap.keySet().iterator();
		while (iter.hasNext()) {
			String key = iter.next();
			Pair value = FeatureIndexMap.get(key);
			System.out.println("("+key+"  "+value.ID+")");
		}
	}
	
	public Map<String, Integer> getFeatureIndexMap(){
		Map<String, Integer> featureIndexMap = new HashMap<String, Integer>();
		Iterator<String> iter = FeatureIndexMap.keySet().iterator();
		while (iter.hasNext()) {
			String feature = iter.next();
			Pair value = FeatureIndexMap.get(feature);
			featureIndexMap.put(feature, value.ID);
		}
		return featureIndexMap;
	}
	
}
