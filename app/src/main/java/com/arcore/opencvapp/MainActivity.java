package com.arcore.opencvapp;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.utils.Converters;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.Video;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private String TAG = "OpenCVApp";

    private CameraBridgeViewBase cameraBViewB;

    private CascadeClassifier cascadeClassifier;
    private Mat grayscaleImage;
    private int absoluteFaceSize;

    public Mat accumulatedBackground;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        int viewId = R.layout.activity_main;
        setContentView(viewId);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        askPermissions();
        initCamera();
    }

    private void initCamera() {
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraBViewB = new JavaCameraView(this, -1);
        cameraBViewB = findViewById(R.id.ivCamera);

        cameraBViewB.setCvCameraViewListener(this);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Toast.makeText(getBaseContext(), "OpenCV loaded successfully", Toast.LENGTH_SHORT).show();
                initOpenCVVariables();
                System.loadLibrary("opencv_java4");

                try {
                    // load cascade file from application resources
                    InputStream is = getResources().openRawResource(R.raw.haarcascade_fullbody);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    File mCascadeFile = new File(cascadeDir, "haarcascade_fullbody.xml");
                    FileOutputStream os = new FileOutputStream(mCascadeFile);

                    byte[] buffer = new byte[4096];
                    int bytesRead;
                    while ((bytesRead = is.read(buffer)) != -1) {
                        os.write(buffer, 0, bytesRead);
                    }
                    is.close();
                    os.close();

                    cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());

//                    cascadeDir.delete();

                } catch (IOException e) {
                    Toast.makeText(getBaseContext(), "ERROR!", Toast.LENGTH_SHORT).show();
                }

                cameraBViewB.enableView();
            } else {
                super.onManagerConnected(status);
            }
        }
    };

    private void initOpenCVVariables(){
        accumulatedBackground = new Mat();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "NOT OK", Toast.LENGTH_SHORT).show();
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Toast.makeText(this, "OK", Toast.LENGTH_SHORT).show();
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);
        absoluteFaceSize = (int) (height * 0.2);
    }

    @Override
    public void onCameraViewStopped() {
    }



    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat inputMat = inputFrame.rgba();
        Mat grayImage = inputFrame.gray();
        /*
                                        Выделение области стоящего человека зелёным прямоугольником
        Imgproc.cvtColor(inputMat, grayscaleImage, Imgproc.COLOR_RGBA2RGB);
        MatOfRect faces = new MatOfRect();
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 2, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i <facesArray.length; i++)
            Imgproc.rectangle(inputMat, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);*/

//        inputMat = doCanny(inputMat, grayImage);
//        return convert(inputMat);
        return process(inputMat);
    }


    private Mat doCanny(Mat frame, Mat grayImage){
        Mat detectedEdges = new Mat();
        Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(grayImage, detectedEdges, new Size(15, 15));
        int threshold = 10;
        Imgproc.Canny(detectedEdges, detectedEdges, threshold, threshold * 3, 3, false);
/*
        Mat dest = new Mat();
        Core.add(dest, Scalar.all(0), dest);
        frame.copyTo(dest, detectedEdges);*/

        return detectedEdges;
    }
    
    public static Mat convert(Mat mat){
        Scalar white = new Scalar(255, 255, 255);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat dest = Mat.zeros(mat.size(), CvType.CV_8UC3);
        Imgproc.findContours(mat, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(dest, contours, -1, white);

        return dest;
    }

    /**
     * @param inputImage
     * threshold - коэффициент double; Чем выше число, тем выше контраст ч/б
     * learningRate - коэффициент double
     * @return
     */
    private double learningRate = 0.9;
    private double threshold = 30;
    private int frameCounter = 0;

    private  Mat process(Mat inputImage){
        Mat foregroundThresh = new Mat();
        Mat inputGray = new Mat();
        Mat backImage = new Mat();
        Mat foreground = new Mat();
        Mat inputFloating = new Mat();

        Imgproc.cvtColor(inputImage, inputGray, Imgproc.COLOR_BGR2GRAY);
        if(accumulatedBackground.empty()) {
            inputGray.convertTo(accumulatedBackground, CvType.CV_32F);
        }
        accumulatedBackground.convertTo(backImage, CvType.CV_8U);
        Core.absdiff(backImage, inputGray, foreground);
        Imgproc.threshold(foreground, foregroundThresh, threshold, 255, Imgproc.THRESH_BINARY_INV);
        inputGray.convertTo(inputFloating, CvType.CV_32F);
        Imgproc.accumulateWeighted(inputFloating, accumulatedBackground, learningRate, foregroundThresh);
        if(frameCounter == 1) {
            inputGray.convertTo(accumulatedBackground, CvType.CV_32F);
            frameCounter = 0;
        } else {
            frameCounter++;
        }
        return negative(foregroundThresh);
    }

    private Mat negative(Mat foregroundThresh){
        Mat result = new Mat();
        Mat white = foregroundThresh.clone();
        white.setTo(new Scalar(255.0));
        Core.subtract(white, foregroundThresh, result);
        return fastDenoising(result);
    }

    public Mat fastDenoising(Mat image) {
        Mat result = new Mat();
        Mat tmp = new Mat();

        Mat kernel = new Mat(new Size(3, 3), CvType.CV_8UC1, new Scalar(255));
        Imgproc.morphologyEx(image, tmp, Imgproc.MORPH_OPEN,kernel);
        Imgproc.morphologyEx(tmp, result, Imgproc.MORPH_CLOSE,kernel);
        return result;
    }

    private void askPermissions() {
        String permission = Manifest.permission.CAMERA;
        int grant = ContextCompat.checkSelfPermission(this, permission);
        if (grant != PackageManager.PERMISSION_GRANTED) {
            String[] permission_list = new String[1];
            permission_list[0] = permission;
            ActivityCompat.requestPermissions(this, permission_list, 1);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "permission granted", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "permission not granted", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (cameraBViewB != null)
            cameraBViewB.disableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (cameraBViewB != null)
            cameraBViewB.disableView();
    }

}