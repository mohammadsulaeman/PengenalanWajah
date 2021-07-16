package com.example.pengenalanwajah;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;


import android.graphics.Bitmap;
import android.os.Build;
import android.os.Environment;
import android.text.InputType;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import com.facebook.stetho.Stetho;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.face.Face;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import static org.opencv.objdetect.Objdetect.CASCADE_SCALE_IMAGE;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static String TAG = MainActivity.class.getSimpleName();
    private CameraBridgeViewBase openCVCamera;
    private Mat rgba,gray;
    private CascadeClassifier classifier;
    private MatOfRect faces;
    private static final int PERMS_REQUEST_CODE = 123;
    private ArrayList<Mat> images;
    private ArrayList<String> imagesLabels;
    private Storage local;
    private String[] uniqueLabels;
    private LinearLayout layoutfooter;
    private ListView mListView;
    private EditText edtNama;
    private Button submit,detect;
    private TextView headertext;
    FaceRecognizer recognize;
    private boolean trainfaces() {
        if(images.isEmpty())
            return false;
        List<Mat> imagesMatrix = new ArrayList<>();
        for (int i = 0; i < images.size(); i++)
            imagesMatrix.add(images.get(i));
        Set<String> uniqueLabelsSet = new HashSet<>(imagesLabels); // Get all unique labels
        uniqueLabels = uniqueLabelsSet.toArray(new String[uniqueLabelsSet.size()]); // Convert to String array, so we can read the values from the indices

        int[] classesNumbers = new int[uniqueLabels.length];
        for (int i = 0; i < classesNumbers.length; i++)
            classesNumbers[i] = i + 1; // Create incrementing list for each unique label starting at 1
        int[] classes = new int[imagesLabels.size()];
        for (int i = 0; i < imagesLabels.size(); i++) {
            String label = imagesLabels.get(i);
            for (int j = 0; j < uniqueLabels.length; j++) {
                if (label.equals(uniqueLabels[j])) {
                    classes[i] = classesNumbers[j]; // Insert corresponding number
                    break;
                }
            }
        }
        Mat vectorClasses = new Mat(classes.length, 1, CvType.CV_32SC1); // CV_32S == int
        vectorClasses.put(0, 0, classes); // Copy int array into a vector

        recognize = LBPHFaceRecognizer.create(3,8,8,8,200);
        recognize.train(imagesMatrix, vectorClasses);
        if(SaveImage())
            return true;

        return false;
    }
    public void showLabelDialog(){
        Set<String> uniqueLabelsSet = new HashSet<>(imagesLabels); // Get all unique labels
        if (!uniqueLabelsSet.isEmpty()){
            String[] uniqueLabels = uniqueLabelsSet.toArray(new String[uniqueLabelsSet.size()]); // Convert to String array for ArrayAdapter
            Arrays.sort(uniqueLabels); // Sort labels alphabetically
            final ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, uniqueLabels) {
                @Override
                public @NonNull
                View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
                    TextView textView = (TextView) super.getView(position, convertView, parent);
                    if (getResources().getBoolean(R.bool.isTablet))
                        textView.setTextSize(20); // Make text slightly bigger on tablets compared to phones
                    else
                        textView.setTextSize(18); // Increase text size a little bit
                    return textView;
                }
            };
            mListView.setVisibility(View.VISIBLE);
            headertext.setVisibility(View.VISIBLE);
            headertext.setText(getString(R.string.selectnama));
            mListView.setAdapter(arrayAdapter);
            mListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                @Override
                public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                    addLabel(arrayAdapter.getItem(position));
                    Log.i(TAG, "Labels Size "+imagesLabels.size()+"");
                    Intent main = new Intent(MainActivity.this,MainActivity.class);
                    main.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
                    startActivity(main);
                    finish();
                }
            });
        }else {
            showEnterLabelDialog();
        }
    }
    public void showEnterLabelDialog(){
        headertext.setVisibility(View.VISIBLE);
        headertext.setText(getString(R.string.isinama));
        edtNama.setVisibility(View.VISIBLE);
        submit.setVisibility(View.VISIBLE);
        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String input = edtNama.getText().toString();
                if (!input.isEmpty()){
                    addLabel(input);
                    edtNama.setVisibility(View.GONE);
                    headertext.setVisibility(View.GONE);
                    detect.setText(getString(R.string.detect));
                    submit.setVisibility(View.GONE);

                }
            }
        });
    }
    private void addLabel(String string) {
        String label = string.substring(0, 1).toUpperCase(Locale.US) + string.substring(1).trim().toLowerCase(Locale.US); // Make sure that the name is always uppercase and rest is lowercase
        imagesLabels.add(label); // Add label to list of labels
        Log.i(TAG, "Label: " + label);

    }
    public boolean SaveImage() {
        File path = new File(Environment.getExternalStorageDirectory(), "TrainedFace");
        path.mkdirs();
        String filename = "lbph_trained_data.xml";
        File file = new File(path, filename);
        recognize.save(file.toString());
        if(file.exists())
            return true;
        return false;
    }
    public void cropedImages(Mat mat) {
        Rect rect_Crop=null;
        for(Rect face: faces.toArray()) {
            rect_Crop = new Rect(face.x, face.y, face.width, face.height);
        }
        Mat croped = new Mat(mat, rect_Crop);
        images.add(croped);
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        Stetho.initializeWithDefaults(this);

        if (hasPermissions()){
            Toast.makeText(this, getString(R.string.izindiberikan), Toast.LENGTH_SHORT).show();
            Log.i(TAG, "Permission Granted Before");

        }
        else {
            requestPerms();
        }

        openCVCamera = (CameraBridgeViewBase)findViewById(R.id.java_camera_view);
        openCVCamera.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        openCVCamera.setVisibility(SurfaceView.VISIBLE);
        openCVCamera.setCvCameraViewListener(this);
        local = new Storage(this);
        mListView = findViewById(R.id.listdata);
        edtNama = findViewById(R.id.editNamakry);
        submit = findViewById(R.id.submitdata);
        headertext = findViewById(R.id.textnama);
        detect = findViewById(R.id.take_picture_button);
        detect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(gray.total() == 0)
                    Toast.makeText(getApplicationContext(), getString(R.string.tidakdapatmendeteksiwajah), Toast.LENGTH_SHORT).show();
                classifier.detectMultiScale(gray,faces,1.1,3,0|CASCADE_SCALE_IMAGE, new Size(30,30));
                if(!faces.empty()) {
                    if(faces.toArray().length > 1)
                        Toast.makeText(getApplicationContext(), getString(R.string.duolarang), Toast.LENGTH_SHORT).show();
                    else {
                        if(gray.total() == 0) {
                            Log.i(TAG, "Empty gray image");
                            return;
                        }
                        cropedImages(gray);
                        showLabelDialog();
                        Toast.makeText(getApplicationContext(), getString(R.string.wajahdetect), Toast.LENGTH_SHORT).show();
                    }
                }else
                    Toast.makeText(getApplicationContext(), getString(R.string.wajahtakkenal), Toast.LENGTH_SHORT).show();
            }
        });
    }
    @SuppressLint("WrongConstant")
    private boolean hasPermissions(){
        int res = 0;
        //string array of permissions,
        String[] permissions = new String[]{Manifest.permission.CAMERA};

        for (String perms : permissions){
            res = checkCallingOrSelfPermission(perms);
            if (!(res == PackageManager.PERMISSION_GRANTED)){
                return false;
            }
        }
        return true;
    }
    private void requestPerms(){
        String[] permissions = new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE,Manifest.permission.READ_EXTERNAL_STORAGE};
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            requestPermissions(permissions,PERMS_REQUEST_CODE);

        }
    }
    @SuppressLint("MissingSuperCall")
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        boolean allowed = true;
        switch (requestCode){
            case PERMS_REQUEST_CODE:
                for (int res : grantResults){
                    // if user granted all permissions.
                    allowed = allowed && (res == PackageManager.PERMISSION_GRANTED);
                }
                break;
            default:
                // if user not granted permissions.
                allowed = false;
                break;
        }
        if (allowed){
            //user granted all permissions we can perform our task.
            Log.i(TAG, "Permission has been added");
        }
        else {
            // we will give warning to user that they haven't granted permissions.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) || shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE) ||
                        shouldShowRequestPermissionRationale(Manifest.permission.READ_EXTERNAL_STORAGE)){
                    Toast.makeText(this, getString(R.string.tolakijin), Toast.LENGTH_SHORT).show();
                }
            }
        }
    }
    private BaseLoaderCallback callbackLoader = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch(status) {
                case BaseLoaderCallback.SUCCESS:
                    faces = new MatOfRect();
                    openCVCamera.enableView();

                    images = local.getListMat("images");
                    imagesLabels = local.getListString("imagesLabels");

                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onPause() {
        super.onPause();
        if(openCVCamera != null)
            openCVCamera.disableView();


    }
    @Override
    protected void onStop() {
        super.onStop();
        if (images != null && imagesLabels != null) {
            local.putListMat("images", images);
            local.putListString("imagesLabels", imagesLabels);
            Log.i(TAG, "Images have been saved");
            if (trainfaces()){
                images.clear();
                imagesLabels.clear();
            }
        }
    }
    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(openCVCamera != null)
            openCVCamera.disableView();
    }
    @Override
    protected void onResume(){
        super.onResume();
        if(OpenCVLoader.initDebug()) {
            Log.i(TAG, "System Library Loaded Successfully");
            callbackLoader.onManagerConnected(BaseLoaderCallback.SUCCESS);
        } else {
            Log.i(TAG, "Unable To Load System Library");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, callbackLoader);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        rgba = new Mat();
        gray = new Mat();
        classifier = FileUtils.loadXMLS(this, "lbpcascade_frontalface_improved.xml");
    }

    @Override
    public void onCameraViewStopped() {
        rgba.release();
        gray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat mGrayTmp = inputFrame.gray();
        Mat mRgbaTmp = inputFrame.rgba();

        int orientation = openCVCamera.getScreenOrientation();
        if (openCVCamera.isEmulator()) // Treat emulators as a special case
            Core.flip(mRgbaTmp, mRgbaTmp, 1); // Flip along y-axis
        else {
            switch (orientation) { // RGB image
                case ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT:
                    Core.flip(mRgbaTmp, mRgbaTmp, 0); // Flip along x-axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE:
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE:
                    Core.flip(mRgbaTmp, mRgbaTmp, 1); // Flip along y-axis
                    break;
            }
            switch (orientation) { // Grayscale image
                case ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:
                    Core.transpose(mGrayTmp, mGrayTmp); // Rotate image
                    Core.flip(mGrayTmp, mGrayTmp, -1); // Flip along both axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT:
                    Core.transpose(mGrayTmp, mGrayTmp); // Rotate image
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE:
                    Core.flip(mGrayTmp, mGrayTmp, 1); // Flip along y-axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE:
                    Core.flip(mGrayTmp, mGrayTmp, 0); // Flip along x-axis
                    break;
            }
        }
        gray = mGrayTmp;
        rgba = mRgbaTmp;
        Imgproc.resize(gray, gray, new Size(200,200.0f/ ((float)gray.width()/ (float)gray.height())));
        return rgba;
    }
}