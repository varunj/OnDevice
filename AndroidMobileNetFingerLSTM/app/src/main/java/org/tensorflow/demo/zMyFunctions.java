package org.tensorflow.demo;

import android.widget.Button;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;


public class zMyFunctions {

    static String[] CLASSES = {"up","down","left","right","star","del","square","carret","tick","circlecc"};


    public static int findArgMax(float[] mat1) {
        int ans = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i=0; i<mat1.length; i++) {
            float elem = mat1[i];
            if (elem > max) {
                max = elem;
                ans = i;
            }
        }
        return ans;
    }

    public static float[] interpolateTo200andReorder(ArrayList<ArrayList<Float>> inp, int TARGET_LEN) {
        float[] out = new float[400];

        // interpolate <200 to 200
        if (inp.size() < TARGET_LEN) {
        	int x = 1;
        	while (inp.size() < TARGET_LEN) {
        		ArrayList<Float> temp = new ArrayList<Float>();
        		temp.add((inp.get(x-1).get(0)+inp.get(x).get(0))/2);
        		temp.add((inp.get(x-1).get(1)+inp.get(x).get(1))/2);
        		inp.add(x, temp);
        		x = x + 2;
        		if (x == inp.size()) {
        			x = 1;
        		}
        	}
        }

        // copy to double[][]
        int i = 0;
        for (ArrayList<Float> x : inp) {
            out[i] = x.get(0);
            out[i+1] = x.get(1);
            i = i + 2;
        }

        return out;
    }
}