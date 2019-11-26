/*
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 */

package peersim.gossip;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.eval.ROCBinary;

import java.util.Arrays;
import java.lang.Integer;

import java.io.FileReader;

import java.io.LineNumberReader;
import peersim.gossip.PegasosNode;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import peersim.config.Configuration;
import peersim.config.FastConfig;
import peersim.core.*;
import peersim.cdsim.*;


import java.net.MalformedURLException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.text.ParseException;
import java.io.BufferedReader;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 *  @author Nitin Nataraj
 */


public class GadgetProtocol implements CDProtocol {
	private static final String PAR_LAMBDA = "lambda";
	private static final String PAR_ITERATION = "iter";
	public static boolean flag = false;
	public static int t = 0;
	public static boolean optimizationDone = false;	
	public double EPSILON_VAL = 0.01;
	protected int lid;
	protected double lambda;
	protected int T;
	public static double[][] optimalB;
	public static int end = 0;
	public static boolean pushsumobserverflag = false;
	public static final int CONVERGENCE_COUNT = 10;
	private String protocol;
	private String resourcepath;


	/**
	 * Default constructor for configurable objects.
	 */
	public GadgetProtocol(String prefix) {
		lid = FastConfig.getLinkable(CommonState.getPid());
		
		protocol = Configuration.getString(prefix + "." + "prot", "pushsum1");
		
	}

	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverRequest(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}
	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverResponse(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}

	/**
	 * Clone an existing instance. The clone is considered 
	 * new, so it cannot participate in the aggregation protocol.
	 */
	public Object clone() {
		GadgetProtocol gp = null;
		try { gp = (GadgetProtocol)super.clone(); }
		catch( CloneNotSupportedException e ) {} // never happens
		return gp;
	}
	
	
	protected List<Node> getPeers(Node node) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) {
			List<Node> l = new ArrayList<Node>(linkable.degree());			
			for(int i=0;i<linkable.degree();i++) {
				l.add(linkable.getNeighbor(i));
			}
			return l;
		}
		else
			return null;						
	}			


	public void nextCycle(Node node, int pid) {
		
		int iter = CDState.getCycle(); // Gets the current cycle of Gadget
		PegasosNode pn = (PegasosNode)node; // Initializes the Pegasos Node
		
		final String resourcepath = pn.resourcepath;
		String csv_filename = resourcepath + "/run" + pn.numRun + "/vpnn_results_temp.csv";
		
		
			System.out.println("Training on node: " + pn.getID());
			INDArray features = pn.trainSet.getFeatures();
			INDArray labels = pn.trainSet.getLabels();
			
			INDArray testLabels = pn.testSet.getLabels();
			INDArray testFeatures = pn.testSet.getFeatures();
			
			INDArray features_batch;
			INDArray labels_batch;
			int numBatches = 20;
			int batchSize = (int)pn.trainSet.getFeatures().size(0)/ numBatches;
			
			System.out.println("Num batches: " + numBatches + " Batch size: " + batchSize);
			for (int batch = 0; batch < numBatches; batch++) {
	            
	            int start_index = batchSize*batch;
	            int end_index = batchSize*(batch+1);
	            
	           
	            features_batch = features.get(NDArrayIndex.interval(start_index,end_index));
	            labels_batch = labels.get(NDArrayIndex.interval(start_index,end_index));
	            //System.out.println(features_batch);
	            
	            pn.model.setInput(features_batch);
	            List<INDArray> output = pn.model.feedForward(true, false);
	            INDArray predictions = output.get(output.size() - 1);
	
	            INDArray diff = labels_batch.sub(predictions);
	            INDArray externalError = diff.mul(diff).mul(0.5);
	
	            Pair<Gradient, INDArray> p = pn.model.backpropGradient(externalError, null);  //Calculate backprop gradient based on error array
	
	            Gradient gradient = p.getFirst();
	            int iteration = 0;
	            int epoch = 0;
	            pn.model.getUpdater().update(pn.model, gradient, iteration, epoch, 
	            		(int)pn.trainSet.getFeatures().size(0), LayerWorkspaceMgr.noWorkspaces());
	
	            INDArray updateVector = gradient.gradient();
	            INDArray a = pn.model.params().addi(updateVector);
			}
	
			
			   
	        ///////////////////////////////////////////////////////////////////////////////////  
	            
			if (Network.size() > 1) {
			// Select a neighbor
			PegasosNode peer = (PegasosNode)selectNeighbor(node, pid);
		    System.out.println("Node "+pn.getID()+" is gossiping with Node "+peer.getID()+"....");
			
		    
		    INDArray features2;
	        features2 = peer.trainSet.getFeatures();
	        // labels will be the same as first node
	        
	        
	        
	        /*
	        for (int batch = 0; batch < numBatches; batch++) {
	            
	            int start_index = batchSize*batch;
	            int end_index = batchSize*(batch+1);
	            
	           
	            features_batch = features2.get(NDArrayIndex.interval(start_index,end_index));
	            labels_batch = labels.get(NDArrayIndex.interval(start_index,end_index));
	            //System.out.println(features2_batch);
	            
	            peer.model.setInput(features_batch);
	            List<INDArray> output2 = peer.model.feedForward(true, false);
	            INDArray predictions2 = output2.get(output2.size() - 1);
	
	            INDArray diff2 = labels_batch.sub(predictions2);
	            INDArray externalError2 = diff2.mul(diff2);
	
	            Pair<Gradient, INDArray> p = peer.model.backpropGradient(externalError2, null);  //Calculate backprop gradient based on error array
	
	            Gradient gradient = p.getFirst();
	            int iteration = 0;
	            int epoch = 0;
	            peer.model.getUpdater().update(peer.model, gradient, iteration, epoch, 
	            		(int)peer.trainSet.getFeatures().size(0), LayerWorkspaceMgr.noWorkspaces());
	
	            INDArray updateVector = gradient.gradient();
	            INDArray a = peer.model.params().addi(updateVector);
	      	}
	        */
	        
	        
	        // Get overall errors
	        
	        pn.model.setInput(features);
	        List<INDArray> output = pn.model.feedForward(true, false);
	        INDArray predictions = output.get(output.size() - 1);
	        
			peer.model.setInput(features2);
	        List<INDArray> output2 = peer.model.feedForward(true, false);
	        INDArray predictions2 = output2.get(output2.size() - 1);
	        
	        // Get average predictions (o1 + o2)/2
	        INDArray averagePredictions = predictions.add(predictions2).div(2); 
	        
	        // Compute average loss
	        INDArray diff = labels.sub(averagePredictions);
	        INDArray combinedAverageError = diff.mul(diff).mul(0.5);
	        
	        
	        // Gossip and backpropagate
	        
	        // Update Model 1
	        Pair<Gradient, INDArray> p = pn.model.backpropGradient(combinedAverageError, null);  //Calculate backprop gradient based on error array
	
	        Gradient gradient = p.getFirst();
	        int iteration = 0;
	        int epoch = 0;
	        pn.model.getUpdater().update(pn.model, gradient, iteration, epoch, 
	        		(int)pn.trainSet.getFeatures().size(0), LayerWorkspaceMgr.noWorkspaces());
	
	        INDArray updateVector = gradient.gradient();
	        INDArray a = pn.model.params().addi(updateVector);    
	        
	        
	        // Update Model 2
	        p = peer.model.backpropGradient(combinedAverageError, null);  //Calculate backprop gradient based on error array
	
	        gradient = p.getFirst();
	        iteration = 0;
	        epoch = 0;
	        peer.model.getUpdater().update(peer.model, gradient, iteration, epoch, 
	        		(int)peer.trainSet.getFeatures().size(0), LayerWorkspaceMgr.noWorkspaces());
	
	        updateVector = gradient.gradient();
	        a = peer.model.params().addi(updateVector);
	        
	        
	        
	        /*
	        // Computing loss
	        pn.model.setInput(features);
	        output = pn.model.feedForward(true, false);
	        predictions = output.get(output.size() - 1);
	        diff = labels.sub(predictions);
	        INDArray externalError = diff.mul(diff);
	        double loss1 = (Double) externalError.sumNumber();
	        
	        peer.model.setInput(features2);
	        output2 = peer.model.feedForward(true, false);
	        predictions2 = output2.get(output2.size() - 1);
	        
	        INDArray diff2 = labels.sub(predictions2);
	        INDArray externalError2 = diff2.mul(diff2);
	        double loss2 = (Double) externalError2.sumNumber();
	        
	        */
			}
	        // Write to file
	        System.out.println("Storing in " + csv_filename);
			
	        INDArray preds = Nd4j.zeros(labels.size(0), Network.size());
	        INDArray predsTest = Nd4j.zeros(pn.testSet.getFeatures().size(0), Network.size());
	        INDArray predsProbability = Nd4j.zeros(labels.size(0), Network.size());
	        INDArray predsTestProbability = Nd4j.zeros(pn.testSet.getFeatures().size(0), Network.size());	        
	        
	        INDArray losses = Nd4j.zeros(1, Network.size());
	        INDArray lossesTest = Nd4j.zeros(1, Network.size());
	        double overall_loss = 0.0;
	        try {
				
		        BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename, true));
		        
		        for(int i=0; i < Network.size(); i++) {
		        	
		        	PegasosNode temp_node = (PegasosNode)Network.get(i);
		        	INDArray temp_features = temp_node.trainSet.getFeatures();
		        	INDArray temp_labels = labels;
		        	INDArray tempFeaturesTest = temp_node.testSet.getFeatures();
					INDArray tempLabelsTest = temp_node.testSet.getLabels();
		        	temp_node.model.setInput(temp_features);
					
			        List <INDArray> temp_output = temp_node.model.feedForward(true, false);
			        INDArray temp_predictions = temp_output.get(temp_output.size() - 1);
			        INDArray temp_diff = temp_labels.sub(temp_predictions);
//			        INDArray temp_externalError = crossEntropyLoss(temp_predictions, labels);
			        INDArray temp_externalError = temp_diff.mul(temp_diff);
			        BooleanIndexing.replaceWhere(temp_externalError,0.0, Conditions.isNan());
			        double temp_loss = (Double) temp_externalError.sumNumber();

		        	temp_node.model.setInput(tempFeaturesTest);
			        List <INDArray> temp_output_test = temp_node.model.feedForward(true, false);		        	
			        INDArray temp_predictions_test = temp_output_test.get(temp_output_test.size() - 1);
			        INDArray temp_diff_test = tempLabelsTest.sub(temp_predictions_test);
			        INDArray temp_externalError_test = temp_diff_test.mul(temp_diff_test);
//			        INDArray temp_externalError_test = crossEntropyLoss(temp_predictions_test, tempLabelsTest);
//			        System.out.println("External errors");
//			        System.out.println(temp_externalError_test);
			        BooleanIndexing.replaceWhere(temp_externalError_test,0.0, Conditions.isNan());
			        double temp_loss_test = (Double) temp_externalError_test.sumNumber();

			        ROCBinary rocTrain = new ROCBinary(0);
			        rocTrain.eval(temp_labels, temp_predictions);
			        double auc_train = rocTrain.calculateAUC(0);
			        
//			        System.out.println(temp_loss_test);
			        ROCBinary rocTest = new ROCBinary(0);
			        rocTest.eval(tempLabelsTest, temp_predictions_test);
			        double auc_test = rocTest.calculateAUC(0);
			        System.out.println(auc_test);			        
			        
			        
//			        INDArray int_preds = temp_predictions.cond(Conditions.greaterThanOrEqual(0.5));
//			        INDArray int_preds_test = temp_predictions_test.cond(Conditions.greaterThanOrEqual(0.5));
//			        
//			        INDArray accuracy_vector = int_preds.sub(labels).cond(Conditions.equals(0));
//			        double accuracy = (Double)accuracy_vector.sum(0).div(labels.size(0)).getDouble(0,0) * 100;
//			        
//			        INDArray accuracy_vector_test = int_preds_test.sub(tempLabelsTest).cond(Conditions.equals(0));
//			        double accuracy_test = (Double)accuracy_vector_test.sum(0).div(tempLabelsTest.size(0)).getDouble(0,0) * 100;
//			        
//			        
			        String opString = iter + "," + i + "," + temp_loss + "," + auc_train + "," + temp_loss_test + "," + auc_test + "," + temp_predictions.amean() + "," + temp_predictions_test.amean();
			        System.out.println(opString);
			        bw.write(opString);
					bw.write("\n");
//			        preds.putColumn(i, int_preds);
			        predsProbability.putColumn(i, temp_predictions);
//			        predsTest.putColumn(i,  int_preds_test);
			        predsTestProbability.putColumn(i, temp_predictions_test);
			        losses.putScalar(i, temp_loss);
			        lossesTest.putScalar(i, temp_loss_test);
		        	
		        }
		       
		        INDArray overallPreds = predsProbability.mean(1);
		        INDArray overallPredsTest = predsTestProbability.mean(1);
		        
		        ROCBinary rocTrainOverall = new ROCBinary(0);
		        rocTrainOverall.eval(labels, overallPreds);
		        
		        
		        ROCBinary rocTestOverall = new ROCBinary(0);
		        rocTestOverall.eval(testLabels, overallPredsTest);
		        
		        
		        double overallAUCTrain = rocTrainOverall.calculateAUC(0);
		        double overallAUCTest = rocTestOverall.calculateAUC(0);
		
//		        INDArray voted_preds = preds.mean(1).cond(Conditions.greaterThanOrEqual(0.5));
//		        voted_preds = voted_preds.eq(labels).sum(0).div(labels.size(0));
//		        double overall_accuracy = voted_preds.getDouble(0,0)*100;
//		        
//		        
//		        INDArray voted_preds_test = predsTest.mean(1).cond(Conditions.greaterThanOrEqual(0.5));
//		        System.out.println(tempLabelsTest.size(0));
//		        System.out.println(voted_preds_test.sumNumber());
//
//		        voted_preds_test = voted_preds_test.eq(tempLabelsTest).sum(0).div(tempLabelsTest.size(0));
//		        double overall_accuracy_test = voted_preds_test.getDouble(0,0)*100;
//
//		        
		        String opString =  iter + "," +"Overall" + ","  + losses.mean(1) + "," + overallAUCTrain + "," + lossesTest.mean(1) + "," + overallAUCTest + "," + overallPreds.amean() + "," + overallPredsTest.amean();;
		        System.out.println(opString);
		        bw.write(opString);
				bw.write("\n");
		        bw.close();
		        
		        
	        	}
			catch(Exception e) {System.out.println(e);}
			}
	        /*
	        INDArray temp;
	        temp = preds.get(0);
	        for(int i=1; i < Network.size(); i++) {
	        	temp.add(preds.get(i));
	        }
	        temp = temp.div(Network.size());
	        */
	        
			
//		}     
//		
//		
//		else {
//			
//			System.out.println("Training on node: " + pn.getID());
//			INDArray features = pn.trainSet.getFeatures();
//			INDArray labels = pn.trainSet.getLabels();
//			INDArray features_batch;
//			INDArray labels_batch;
//			int batchSize = pn.batch_size;
//			int numBatches = (int)pn.trainSet.getFeatures().size(0)/ batchSize;
//			
//			System.out.println("Num batches: " + numBatches + " Batch size: " + batchSize);
//			for (int batch = 0; batch < numBatches; batch++) {
//	            
//	            int start_index = batchSize*batch;
//	            int end_index = batchSize*(batch+1);
//	           
//	            features_batch = features.get(NDArrayIndex.interval(start_index,end_index));
//	            labels_batch = labels.get(NDArrayIndex.interval(start_index,end_index));
//	            //System.out.println(features_batch);
//	            
//	            pn.model.setInput(features_batch);
//	            List<INDArray> output = pn.model.feedForward(true, false);
//	            INDArray predictions = output.get(output.size() - 1);
//	            INDArray int_preds = predictions.cond(Conditions.greaterThanOrEqual(0.5));
//	            INDArray diff = labels_batch.sub(int_preds);
//	            INDArray externalError = diff.mul(diff).mul(0.5);
//	            //INDArray externalError = labels_batch.mul(Transforms.log(predictions, true)).mul(-1);
//	            //INDArray externalError = predictions.sub(labels_batch);
//	            Pair<Gradient, INDArray> p = pn.model.backpropGradient(externalError, null);  //Calculate backprop gradient based on error array
//	 
//	            Gradient gradient = p.getFirst();
//	            int iteration = 0;
//	            int epoch = 0;
//	            pn.model.getUpdater().update(pn.model, gradient, iteration, epoch, 
//	            		(int)pn.trainSet.getFeatures().size(0), LayerWorkspaceMgr.noWorkspaces());
//	
//	            INDArray updateVector = gradient.gradient();
//	            INDArray a = pn.model.params().addi(updateVector);
//			}
//			
//			// Computing loss
//	        pn.model.setInput(features);
//	        List <INDArray> output = pn.model.feedForward(true, false);
//	        INDArray predictions = output.get(output.size() - 1);
//	        //System.out.println(predictions);
//	        INDArray int_preds = predictions.cond(Conditions.greaterThanOrEqual(0.5));
//	        INDArray diff = labels.sub(int_preds);
//	        INDArray externalError = diff.mul(diff).mul(0.5);
//	        //INDArray externalError = labels.mul(Transforms.log(predictions, true)).mul(-1);
//	        double loss1 = (Double) externalError.sumNumber();
//			
//	        
//	        INDArray accuracy_vector = int_preds.sub(labels).cond(Conditions.equals(0));
//	        double accuracy = (Double)accuracy_vector.sum(0).div(labels.size(0)).getDouble(0,0) * 100;
//	        
//	        
//	        System.out.println("Computing test loss");
//	        // Computing test loss
//	        INDArray features_test = pn.testSet.getFeatures();
//	        INDArray labels_test = pn.testSet.getLabels();
//	        pn.model.setInput(features_test);
//	        List <INDArray> output_test = pn.model.feedForward(true, false);
//	        INDArray predictions_test = output_test.get(output_test.size() - 1);
//	        INDArray int_preds_test = predictions_test.cond(Conditions.greaterThanOrEqual(0.5));
//	        INDArray diff_test = labels_test.sub(int_preds_test);
//	        INDArray externalError_test = diff_test.mul(diff_test).mul(0.5);
//	        double loss1_test = (Double) externalError_test.sumNumber();
//	        
//	        INDArray accuracy_vector_test = int_preds_test.sub(labels_test).cond(Conditions.equals(0));
//	        double accuracy_test = (Double)accuracy_vector_test.sum(0).div(labels_test.size(0)).getDouble(0,0) * 100;
//	        
//	        // Write to file
//	        
//	        String opString =  iter + "," + pn.getID() + "," + loss1 +"," + accuracy+","+loss1_test +"," +accuracy_test;
//	        //System.out.println(opString);
//	        System.out.println("Storing in " + csv_filename);
//			
//	        try {
//				BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename, true));
//				
//				bw.write(opString);
//				bw.write("\n");
//				bw.close();
//				}
//			catch(Exception e) {}
//	        
//			
//		}
//        	
//	}

	/**
	 * Selects a random neighbor from those stored in the {@link Linkable} protocol
	 * used by this protocol.
	 */
	protected Node selectNeighbor(Node node, int pid) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) 
			return linkable.getNeighbor(
					CommonState.r.nextInt(linkable.degree()));
		else
			return null;
	}

	public static void writeIntoFile(String millis) {
		File file = new File("exec-time.txt");
		 
		// if file doesnt exists, then create it
		if (!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		FileWriter fw;
		try {
			fw = new FileWriter(file.getAbsoluteFile(),true);

		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(millis+"\n");
		bw.close();
		} catch (IOException e)
		
		 {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		

	}
	
	private INDArray crossEntropyLoss(INDArray predictions, INDArray labels) {

		/*
		 Computes the cross-entropy loss between predictions and labels array.
		 */
		int numRows = predictions.rows();
		int numCols = predictions.columns();
		
		INDArray batchLossVector = Nd4j.zeros(numRows, 1);
		for(int i=0;i<numRows;i++) {
			double loss = 0.0;
			for(int j=0;j<numCols;j++) {
				loss += ((labels.getDouble(i,j)) * Math.log(predictions.getDouble(i,j) + 1e-15)) + (
						(1-labels.getDouble(i,j)) * Math.log((1-predictions.getDouble(i,j)) + 1e-15)
						)
						;
				
			}
			batchLossVector.putScalar(i, 0, -loss);
		}
		return batchLossVector;
	}
}

