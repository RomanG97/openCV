package com.arcore.opencvapp;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraManager;
import android.os.Bundle;
import android.provider.Settings;
import android.renderscript.RenderScript;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
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
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import utils.Utils;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private String TAG = "OpenCVApp";

    private CameraBridgeViewBase cameraBViewB;
    private CameraManager cameraManager;

    private Mat white;
    private Mat inputMat;

    public Mat process_accumulatedBackground;

    private Mat process_foregroundThresh;
    private Mat process_inputGray;
    private Mat process_backImage;
    private Mat process_foreground;
    private Mat process_inputFloating;

    private Mat result;


    private Mat fastDenoising_result;
    private Mat fastDenoising_tmp;

    private Mat kernel;

    private Mat simpleCanny_detectedEdges;

    private Mat mdBack;
    private Mat mdReturnMat;

    private Mat mdM;
    private Mat mdBufIm;
    private Mat mdMask;

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
        cameraBViewB = (JavaCameraView) findViewById(R.id.ivCamera);

        cameraBViewB.setCvCameraViewListener(this);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Toast.makeText(getBaseContext(), "OpenCV loaded successfully", Toast.LENGTH_SHORT).show();
                System.loadLibrary("opencv_java4");

                cameraBViewB.enableView();
            } else {
                super.onManagerConnected(status);
            }
        }
    };

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
        process_accumulatedBackground = new Mat();
        white = new Mat();
        inputMat = new Mat();

        process_foregroundThresh = new Mat();
        process_inputGray = new Mat();
        process_backImage = new Mat();
        process_foreground = new Mat();
        process_inputFloating = new Mat();

        result = new Mat();

        fastDenoising_result = new Mat();
        fastDenoising_tmp = new Mat();

        kernel = new Mat(new Size(3, 3), CvType.CV_8UC1, new Scalar(255));

        simpleCanny_detectedEdges = new Mat();

        mdBack = new Mat();
        mdReturnMat = new Mat();

        mdM = new Mat();
        mdBufIm = new Mat();
        mdMask = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
    }



    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        inputMat = inputFrame.gray();

        return movementDetector(inputMat);
    }

    /**
     * @param inputImage
     * MOVING IMAGE SELECTION
     * @return Mat
     */
    private Mat movementDetector(Mat inputImage){
        if(mdBack.empty()){
            inputImage.copyTo(mdBack);
        }
        Core.absdiff(mdBack, inputImage, mdM);
        inputImage.copyTo(mdBack);
        Imgproc.threshold(mdM, mdM, 5, 255, Imgproc.THRESH_BINARY);
        Imgproc.blur(mdM, mdM, new Size(5, 5));
        Imgproc.morphologyEx(mdM, mdM, Imgproc.MORPH_OPEN, new Mat(new Size(5, 5), CvType.CV_8UC1, new Scalar(255)));
        mdBack.convertTo(mdBufIm, CvType.CV_32FC1);
        Imgproc.accumulateWeighted(mdM, mdBufIm, 4.5, mdM);
        int threshold = 170;
        Imgproc.Canny(mdM, mdM, threshold, threshold * 3, 3, false);
        mdMask.release();
        mdMask = new Mat(mdM.size(), CvType.CV_8UC3, new Scalar(0,0,0));
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mdM, contours, new Mat(), Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

        for (int i = 0 ; i < contours.size() ; i++) {
            MatOfPoint2f curContour2f = new MatOfPoint2f(contours.get(i).toArray());
            Imgproc.approxPolyDP(curContour2f, curContour2f, 0.04 * Imgproc.arcLength(curContour2f, true), true);
            contours.set(i, new MatOfPoint(curContour2f.toArray()));
            Imgproc.drawContours(mdMask, contours, i, new Scalar(0, 255, 0), 3);

        }
        return mdMask;
    }

    /**
     * @param foregroundThresh
     * APPLYING NEGATIVE FILTER
     * @return Mat
     */
    private Mat negative(Mat foregroundThresh){
        foregroundThresh.copyTo(white);
        white.setTo(new Scalar(255.0));
        Core.subtract(white, foregroundThresh, result);
        return fastDenoising(result);
    }

    /**
     * @param image
     * FILTER MAPPING TO FILTER THE IMAGE
     * @return Mat
     */
    public Mat fastDenoising(Mat image) {
        Imgproc.morphologyEx(image, fastDenoising_tmp, Imgproc.MORPH_OPEN,kernel);
        Imgproc.morphologyEx(fastDenoising_tmp, fastDenoising_result, Imgproc.MORPH_CLOSE,kernel);
        fastDenoising_tmp.release();
        image.release();
        return simpleCanny(fastDenoising_result);
    }

    /**
     * @param image
     * IMAGE BORDER SELECTION
     * @return Mat
     */
    public Mat simpleCanny(Mat image){
        Imgproc.blur(image, simpleCanny_detectedEdges, new Size(3, 3));
        int threshold = 10;
        Imgproc.Canny(simpleCanny_detectedEdges, simpleCanny_detectedEdges, threshold, threshold * 3, 3, true);
        return convert(simpleCanny_detectedEdges);
    }

    /**
     * @param mat
     * CONVERTS THE CONTOUR TO A LIST OF POINTS
     * @return Mat
     */
    public Mat convert(Mat mat) {
        Scalar white = new Scalar(255, 255, 255);
        List<MatOfPoint> MatmatOfPointList = new  ArrayList<>();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat dest = Mat.zeros(mat.size(), CvType.CV_8UC3);
        Imgproc.findContours(mat, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(dest, contours, -1, white);
//        for(MatOfPoint mop : contours){
//            Rect r = Imgproc.boundingRect(mop);
//            Imgproc.rectangle(dest, new Point(r.x, r.y), new Point(r.x + r.width - 1, r.y + r.height - 1), white);
//            MatOfPoint matOfPoint = new MatOfPoint(new Point(r.x, r.y),
//                    new Point(r.x + r.width - 1, r.y),
//                    new Point(r.x + r.width - 1, r.y + r.height - 1),
//                    new Point(r.x, r.y + r.height - 1));
//            MatmatOfPointList.add(matOfPoint);
//        }

//        Imgproc.fillPoly(dest, MatmatOfPointList, white);

        return dest;
    }


   //Выделяет движущийся объект кривыми по крайним точкам, который закрашивает белым
    public Mat matfit(Mat mat, List<MatOfPoint> contoursList){
        List<MatOfPoint> mOp = new ArrayList<>();
        MatOfInt hull = new MatOfInt();
        for(MatOfPoint contour : contoursList){
            Imgproc.convexHull(contour, hull, false);

            MatOfPoint mopHull = new MatOfPoint();
            mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
            for (int j = 0; j < hull.size().height; j++) {
                int index = (int) hull.get(j, 0)[0];
                double[] point = new double[] { contour.get(index, 0)[0], contour.get(index, 0)[1] };
                mopHull.put(j, 0, point);
            }
            mOp.add(mopHull);
        }
        Mat dest = Mat.zeros(mat.size(), CvType.CV_8UC3);
        Scalar white = new Scalar(255, 255, 255);
        Imgproc.drawContours(dest, mOp, -1, white);
        Imgproc.fillPoly(dest, mOp, white);
        return dest;
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
        if (cameraBViewB != null) {
            cameraBViewB.disableView();
        }
    }

}