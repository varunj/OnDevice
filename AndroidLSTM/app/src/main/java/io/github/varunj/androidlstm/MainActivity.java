package io.github.varunj.androidlstm;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class MainActivity extends AppCompatActivity {

    private TextView text_result;
    private Button button_test_lstm;
    private static String ROOT_PATH = "test_resampled_200";
    private static String[] CLASSES = {"up","down","left","right","star","del","square","carret","tick","circlecc"};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        text_result = (TextView) findViewById(R.id.text_result);
        button_test_lstm = (Button) findViewById(R.id.button_test_lstm);
        text_result.setText("input,\nprediction");

        final TensorFlowInferenceInterface inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "frozen_model.pb");
        final String[] outputNames =  new String[] {"dense/BiasAdd"};
        final float[] outputs = new float[10];

        button_test_lstm.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                try {
                    // read some random test file
                    String[] listOfFiles = getAssets().list(ROOT_PATH);
                    Random randomizer = new Random();
                    String randomFileName = listOfFiles[randomizer.nextInt(listOfFiles.length)];
                    String fileName = ROOT_PATH + "/" + randomFileName;
                    float[] out = readFileFromAssets(getApplicationContext(), fileName);
                    int[] inp_len = {out.length/2};

                    // tensorflow stuff
                    inferenceInterface.feed("Placeholder", out, 1,200,2);
                    inferenceInterface.feed("Placeholder_2", inp_len, 1);
                    inferenceInterface.run(outputNames, true);
                    inferenceInterface.fetch("dense/BiasAdd", outputs);
                    text_result.setText(randomFileName.substring(0,randomFileName.length()-4) + ",\n" + CLASSES[findArgMax(outputs)]);
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

    }

    public static float[] readFileFromAssets(Context context, String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(filename)));
        float[] arr = new float[400];
        int i = 0;
        String mLine = reader.readLine();
        while (mLine != null) {
            arr[i] = Float.valueOf(mLine.split(" ")[0]);
            arr[i+1] = Float.valueOf(mLine.split(" ")[1]);
            mLine = reader.readLine();
            i = i + 2;
        }
        reader.close();
        return arr;
    }

    public static int findArgMax(float[] mat1) {
        int ans = -1;
        float max = Float.NEGATIVE_INFINITY;
        for (int i=0; i<mat1.length; i++) {
            float elem = mat1[i];
            if (elem > max) {
                max = elem;
                ans = i;
            }
        }
        return ans;
    }
}