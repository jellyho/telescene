package org.telescene.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.MediaRecorder;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Holds the screen capture permission for the life of the app and records the screen while the PC records a take,
 * one mp4 per take in the app's Movies folder, beside a json with the wall-clock start and stop in milliseconds.
 * Android wants a foreground service of type mediaProjection to hold a capture.
 */
public class CaptureService extends Service {
    static final String TAG = MainActivity.TAG;
    static final int WIDTH = 2160, HEIGHT = 1080, FPS = 30, BITRATE = 16_000_000;

    static volatile MediaProjection projection;
    private static CaptureService self;
    private static Handler worker;
    private static final Runnable APPLY = new Runnable() { @Override public void run() { apply(); } };
    private static boolean wanted, recording;

    private MediaRecorder recorder;
    private VirtualDisplay display;
    private File file;
    private long started;

    @Override
    public int onStartCommand(Intent intent, int flags, int id) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel("capture", "Take capture", NotificationManager.IMPORTANCE_LOW));
        Notification n = new Notification.Builder(this, "capture").setContentTitle("Telescene records takes")
                .setSmallIcon(android.R.drawable.presence_video_online).build();
        startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        if (projection == null && intent != null) {
            MediaProjectionManager m = getSystemService(MediaProjectionManager.class);
            projection = m.getMediaProjection(intent.getIntExtra("result", 0), intent.getParcelableExtra("data"));
            projection.registerCallback(new MediaProjection.Callback() {
                @Override public void onStop() { Log.i(TAG, "screen capture stopped by the system"); projection = null; }
            }, null);
            HandlerThread t = new HandlerThread("capture");
            t.start();
            worker = new Handler(t.getLooper());
            self = this;
            Log.i(TAG, "screen capture ready");
            worker.post(APPLY);  // the PC may have been recording already
        }
        return START_NOT_STICKY;
    }

    static void setRecording(boolean on) {
        synchronized (CaptureService.class) { wanted = on; }
        Handler w = worker;
        if (w != null) w.post(APPLY);
    }

    private static void apply() {
        boolean on;
        synchronized (CaptureService.class) { on = wanted; }
        if (self == null || on == recording) return;
        if (on) self.begin(); else self.end();
    }

    private void begin() {
        if (projection == null) return;
        try {
            File dir = new File(getExternalFilesDir(Environment.DIRECTORY_MOVIES), "takes");
            dir.mkdirs();
            file = new File(dir, new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date()) + ".mp4");
            recorder = new MediaRecorder(this);
            recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
            recorder.setVideoSize(WIDTH, HEIGHT);
            recorder.setVideoFrameRate(FPS);
            recorder.setVideoEncodingBitRate(BITRATE);
            recorder.setOutputFile(file.getAbsolutePath());
            recorder.prepare();
            display = projection.createVirtualDisplay("telescene-take", WIDTH, HEIGHT, 320,
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, recorder.getSurface(), null, null);
            recorder.start();
            started = System.currentTimeMillis();
            recording = true;
            Log.i(TAG, "capturing the take to " + file);
        } catch (Exception e) {
            Log.e(TAG, "screen capture failed to start", e);
            release();
        }
    }

    private void end() {
        long stopped = System.currentTimeMillis();
        try {
            recorder.stop();
        } catch (Exception e) {
            Log.e(TAG, "screen capture failed to stop", e);
        }
        release();
        try (FileWriter w = new FileWriter(new File(file.getPath().replace(".mp4", ".json")))) {
            w.write("{\"start_ms\": " + started + ", \"stop_ms\": " + stopped + "}\n");
        } catch (Exception e) {
            Log.e(TAG, "could not write the take's times", e);
        }
        Log.i(TAG, "captured " + (stopped - started) / 1000.0 + " s to " + file);
    }

    private void release() {
        if (display != null) display.release();
        if (recorder != null) recorder.release();
        display = null;
        recorder = null;
        recording = false;
    }

    @Override
    public void onDestroy() {
        if (recording) end();
        if (projection != null) projection.stop();
        projection = null;
        self = null;
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) { return null; }
}
