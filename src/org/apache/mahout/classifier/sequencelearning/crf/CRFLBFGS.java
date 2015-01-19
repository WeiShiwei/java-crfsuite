package org.apache.mahout.classifier.sequencelearning.crf;

import riso.numerical.LBFGS;

public class CRFLBFGS {
	
	int m=5;
	int[] iprint ;
	boolean diagco;
	double diag [ ]; 
	double eps, xtol;
	int iflag[] = new int[1];
	
	public CRFLBFGS(){}
	
	public void optimize ( int n ,  double[] x , double f , double[] g ){
		if(diag==null){
			m=5;
			iprint = new int [ 2 ];
			iprint [ 1 -1] = 1;
			iprint [ 2 -1] = 0;
			
			diagco=false;
			diag= new double [ n ];
			eps= 1.0e-5;
			xtol= 1.0e-16;
			
			iflag[0]=0;
		}
		/**************************************A调试**********************************************/
		
//		System.out.println("------TaggerImplReducer------");
//		System.out.println("x详细信息：");
//		for(int hh=0;hh<x.length;hh++){
//			System.out.println("x["+hh+"]	"+x[hh]);
//		}
//		System.out.println();
//		System.out.println("f="+f);
//		
//		System.out.println("g详细信息：");
//		for(int hh=0;hh<g.length;hh++){
//			System.out.println("g["+hh+"]	"+g[hh]);
//		}		

//		System.out.println("diag详细信息：");
//		for(int hh=0;hh<diag.length;hh++){
//			System.out.println("diag["+hh+"]	"+diag[hh]);
//		}
		/**************************************B**********************************************/
		/*lbfgs*/
		try{
			LBFGS.lbfgs ( n , m , x , f , g , diagco , diag , iprint , eps , xtol , iflag );
		}
		catch (LBFGS.ExceptionWithIflag e){
			System.err.println( "Sdrive: lbfgs failed.\n"+e );
			return;
		}
	}

	
}
