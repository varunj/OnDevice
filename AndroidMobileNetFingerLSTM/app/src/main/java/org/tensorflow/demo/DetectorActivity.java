/*
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.lite.demo.R; // Explicit import needed for internal Google builds.

import static android.content.ContentValues.TAG;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/pascal_label_map.pbtxt";



  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;

  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;

  // ---new
  static ArrayList<ArrayList<Float>> inpStream = new ArrayList<>();
  static int counter=0;
  private static int[] intValues = new int[99 * 99];
  private static float[] floatValues = new float[99 * 99 * 3];
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;
  Bitmap finalCrop = null;
  // ---new

  private static String abc="abc";


  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {

    Log.d(abc,"onPreviewSizeChosen called");
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      LOGGER.e("Exception initializing classifier!", e);
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }


    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString = detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });
  }

  OverlayView trackingOverlay;

  @Override
  protected void processImage() {
    Log.d(abc,"processImage called");

    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    // ---new
    ImageView imageView_Out= (ImageView) findViewById(R.id.imageView_Out);
    // ---new

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);

                // ---new
                finalCrop = storeImage(rgbFrameBitmap, location, getApplicationContext());


                // https://developer.android.com/reference/android/graphics/RectF#RectF()
                ArrayList<Float> temp = new ArrayList<>();
                temp.add(480-location.bottom);          // (480-location.bottom) z2, (location.bottom) 5x.
                  // also change this in storeImage()
                temp.add(location.left);                // (location.left) z2, (640-location.left) 5x
                inpStream.add(temp);
                boolean val;
                val=inp_check();
                Log.d(abc,"Val: " + val);
                if (val==true & counter==1)
                {
                  Log.d(abc,"Val: " + "input stream clear");
                  /*inpStream.clear();*/
                  Toast toast =
                          Toast.makeText(
                                  getApplicationContext(), "Stop", Toast.LENGTH_SHORT);
                  toast.show();
                  Trigger_classification();
                  counter=0;
                  inpStream.clear();
                }
                else if (val==true & counter==0)
                {
                  Log.d(abc,"Val: " + "input stream clear");
                  inpStream.clear();
                  Toast toast =
                          Toast.makeText(
                                  getApplicationContext(), "Start", Toast.LENGTH_SHORT);
                  toast.show();
                  counter=1;
                }
                /*System.out.println("xxx1: " + (temp.get(0)) + "," + (temp.get(1)));*/
                // ---new

              }
            }

            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            trackingOverlay.postInvalidate();

            requestRender();
            computingDetection = false;
          }
        });

    // ---new
    imageView_Out.setImageBitmap(finalCrop);
    // ---new
  }
    // ---new
  protected boolean inp_check() {

    int leng=inpStream.size();
    int thresh_value=8;
    double x_sd=1000,y_sd=1000;
     /* Log.d(abc,"inpcheck");*/
     if(leng>=thresh_value)
     {

     /* for (ArrayList<Float> x : inpStream)
      {
          Log.d(abc,"inpcheck: " + (x.get(0)) + "," + (x.get(1)));
      }*/
      /*for(int i=0;i<10000;i++){}*/
      float []numarray_x=new float[thresh_value];
      float []numarray_y=new float[thresh_value];
      for(int j=thresh_value-1,i=leng-1;j>=0;i--,j--)
      {
        numarray_x[j]=inpStream.get(i).get(0);
        numarray_y[j]=inpStream.get(i).get(1);
        /*Log.d(abc,"num_x: " + numarray_x[j]);
        Log.d(abc,"num_y: " + numarray_y[j]);*/
      }
       x_sd=zMyFunctions.calculateSD(numarray_x);
       y_sd=zMyFunctions.calculateSD(numarray_y);
       Log.d(abc,"X_sd: " + x_sd);
       Log.d(abc,"Y_sd: " + y_sd);
     }
     if(x_sd<3.5 & y_sd<3.5)
     {return true;}
      else
     { return false;}

  }// ---new



  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }


  // ---new
  public static float[] detectFingerTip(Bitmap image, Context context) {
    TensorFlowInferenceInterface inferenceInterface = new TensorFlowInferenceInterface (context.getAssets(), "frozen_model_fingertip.pb");
    final String[] outputNames =  new String[] {"output_node0"};
    final float[] outputs = new float[2];

    // Preprocess the image data from 0-255 int to normalized float based on the provided parameters.
    image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
    for (int i = 0; i < intValues.length; ++i) {
      final int val = intValues[i];
      floatValues[i * 3 + 0] = (((val >> 16) & 0xFF)- IMAGE_MEAN)/ IMAGE_STD;
      floatValues[i * 3 + 1] = (((val >> 8) & 0xFF)- IMAGE_MEAN)/ IMAGE_STD;
      floatValues[i * 3 + 2] = ((val & 0xFF)- IMAGE_MEAN)/ IMAGE_STD;
    }
    // tensorflow stuff
    inferenceInterface.feed("conv2d_1_input", floatValues, 1,99,99,3);
    inferenceInterface.run(outputNames, true);
    inferenceInterface.fetch(outputNames[0], outputs);
    System.out.println("xxx4: x: " + outputs[0] + " y: " + outputs[1]);
    return outputs;
  }

  public static Bitmap storeImage(Bitmap image, RectF location, Context context) {
    float[] detectionXY = new float[2];
    File pictureFile = getOutputMediaFile();
    if (pictureFile == null) {
      return null;
    }
    try {
      Bitmap croppedBmp = Bitmap.createBitmap(image, (int)location.left,(int)(480-location.bottom), (int)location.width(), (int)location.height());
      croppedBmp = getResizedBitmap(RotateBitmap(croppedBmp, 90), 99,99);
      detectionXY = detectFingerTip(croppedBmp, context);

//      FileOutputStream fos = new FileOutputStream(pictureFile);
//      croppedBmp.compress(Bitmap.CompressFormat.PNG, 90, fos);
//      fos.close();

      Canvas c = new Canvas(croppedBmp);
//      croppedBmp.setPixel((int) detectionXY[0], (int) detectionXY[0], Color.BLACK);

      Paint paint = new Paint();
      paint.setStyle(Paint.Style.FILL);
      paint.setColor(Color.BLACK);
      c.drawCircle((int) detectionXY[0], (int) detectionXY[0], 9, paint);

      return croppedBmp;

    } catch (Exception e) {}
    return null;
  }

  /** Create a File for saving an image or video */
  public static File getOutputMediaFile(){

    File mediaStorageDir = new File(Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "tensorflow");
    // Create the storage directory if it does not exist
    if (! mediaStorageDir.exists()){
      if (! mediaStorageDir.mkdirs()){
        return null;
      }
    }
    // Create a media file name
    String timeStamp = new SimpleDateFormat("ddMMyy_HHmmssSSS").format(new Date());
    File mediaFile;
    String mImageName="MI_"+ timeStamp +".jpg";
    mediaFile = new File(mediaStorageDir.getPath() + File.separator + mImageName);
    return mediaFile;
  }

  public static Bitmap RotateBitmap(Bitmap source, float angle)
  {
    Matrix matrix = new Matrix();
    matrix.postRotate(angle);
    return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
  }

  public static Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
    int width = bm.getWidth();
    int height = bm.getHeight();
    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;
    // CREATE A MATRIX FOR THE MANIPULATION
    Matrix matrix = new Matrix();
    // RESIZE THE BIT MAP
    matrix.postScale(scaleWidth, scaleHeight);

    // "RECREATE" THE NEW BITMAP
    Bitmap resizedBitmap = Bitmap.createBitmap(
            bm, 0, 0, width, height, matrix, false);
    bm.recycle();
    return resizedBitmap;
  }
  // ---new

}

