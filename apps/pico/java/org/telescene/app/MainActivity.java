package org.telescene.app;

import android.app.Activity;
import android.app.NativeActivity;
import android.content.Intent;
import android.media.projection.MediaProjectionManager;
import android.os.Bundle;
import android.util.Log;

/**
 * The native app, plus screen capture of each take: at start it asks once for permission to capture the screen, and
 * main.c calls {@link #setRecording} whenever the PC starts or stops recording a take. The capture is what the system
 * composites, so passthrough is in it.
 */
public class MainActivity extends NativeActivity {
    static final String TAG = "telescene";
    private static final int ASK = 1;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        if (CaptureService.projection == null) {
            MediaProjectionManager m = (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
            startActivityForResult(m.createScreenCaptureIntent(), ASK);
        }
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != ASK) return;
        if (result != Activity.RESULT_OK || data == null) {
            Log.w(TAG, "screen capture refused: takes are recorded on the PC only");
            return;
        }
        Intent start = new Intent(this, CaptureService.class).putExtra("result", result).putExtra("data", data);
        startForegroundService(start);
    }

    /** Called from main.c on the render thread, on every change of the PC's recording status. */
    public static void setRecording(boolean on) {
        CaptureService.setRecording(on);
    }
}
