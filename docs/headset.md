# The headset app

`apps/pico/` is one native Android app, in C with two small Java classes, built without Gradle. It does the following:
- holds the OpenXR session;
- draws the scene the PC sends over passthrough;
- reads the headset, the controllers and `XR_BD_body_tracking` (24 joints);
- sends them to the PC.

It is tested on a PICO 4 Ultra.

## 1. Prepare the headset (once)

1. **Developer mode.** Settings → General → About: tap the software version until Developer appears. In Developer,
   turn on USB debugging.
2. **Motion Trackers**, for the legs in whole-body POSE. Without them the app still runs, and hybrid mode needs only the
   headset and controllers.
   1. Strap one tracker to each ankle, light up. Keep loose trousers from covering them.
   2. In the Motion Tracker app, unpair old trackers, then press Pair. Hold each tracker's button for 6 s, until it
      flashes red and blue.
   3. Press Calibrate with the headset on. First stand straight with the controllers at your sides, then look down at
      your ankles until the headset sees the trackers.
   4. Calibrate again whenever the headset says so.
3. **Network.** Put the headset and the PC on the same network. The PC must accept TCP and UDP on ports 47800–47804
   (`sudo ufw allow 47800:47804/tcp && sudo ufw allow 47800:47804/udp` if ufw is on).

## 2. Build the app

`setup_toolchain.sh` installs JDK 17, the Android SDK (platform 34, build-tools 34.0.0, NDK 26.3, adb) and the Khronos
OpenXR loader 1.1.49 into `~/.local/share/telescene-android` (or `$TELESCENE_ANDROID`). It needs no root and takes
about 2.5 GB. It asks you to accept the Android SDK licences.

```bash
apps/pico/setup_toolchain.sh
apps/pico/build.sh            # writes apps/pico/build/telescene.apk
```

`build.sh` signs the APK with a debug key it creates next to the toolchain. A prebuilt APK may also be attached to
the GitHub release. You can install it with `apps/pico/install.sh telescene.apk` and skip the toolchain.

## 3. Install

Connect the headset by USB and allow debugging when the headset asks. Then:

```bash
apps/pico/install.sh
```

This installs or updates the app and tells it the PC's address (`debug.telescene.pc`). If the PC has several networks,
set the address yourself: `PC_IP=192.168.0.10 apps/pico/install.sh`. Without an address, the app looks for the PC by
UDP broadcast, which some networks block. The address does not survive a headset reboot, so run the script again
after one. `examples/isaaclab_g1/run.sh` runs it for you whenever a headset is on USB.

Once installed, the headset no longer needs the cable.

## 4. Use

1. Start the PC side (`examples/isaaclab_g1/run.sh`, see [simulation.md](simulation.md)). Isaac takes about a minute to
   start. Meanwhile the app shows its progress.
2. In the headset, open **Telescene** from the app library (Unknown sources).
3. Point a controller at CONNECT and pull the trigger. MIRROR shows your own tracked body instead, with no robot.

The controls are in the [README](../README.md#in-the-headset).

Only one 3D app runs at a time. Close other XR apps first. The PC never starts the app; you open it yourself.

Each take (left Y with `--record`) can also save the headset's view. Android asks for screen-capture consent the first
time. The PICO's own screen recording from its quick menu works too. It saves one eye to
`/sdcard/Movies/ScreenRecording/`.

## Settings

The app reads these Android properties, set with `adb shell setprop`:

| Property | Values | Default |
|---|---|---|
| `debug.telescene.pc` | the PC's IPv4 address | found by broadcast |
| `debug.telescene.input` | `tcp` or `udp`: how body and controllers reach the PC | `tcp` |
| `debug.telescene.mode` | `menu`, or `mirror` to open straight in the mirror | `menu` |

## Troubleshooting

| Symptom | Check |
|---|---|
| CONNECT never shows the robot | The PC's firewall (ports 47800–47804); `debug.telescene.pc` is the PC's current address; both on the same network |
| "Controllers Required" dialog | Wake the controllers. The app also runs with hand tracking |
| Legs do not move in POSE | Calibrate the Motion Trackers. The app log (below) prints `body_valid=24/24` when every joint is tracked |
| POSE turns itself off, the PC log says `the headset came off` | The proximity sensor: wear the headset properly |
| The PC log says `HOLDING LAST INPUT` | Input stopped arriving: Wi-Fi. Try `adb shell setprop debug.telescene.input udp` |
| `adb install` fails with a signature mismatch | An APK from another build key is installed: `adb uninstall org.telescene.app` first |

`adb logcat -s telescene` shows the app's own log.
