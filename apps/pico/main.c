// SPDX-License-Identifier: MIT
// Teleop client for PICO 4 Ultra. One app owns the OpenXR session: it renders the scene locally, so
// head motion never crosses the network, and it sends body and controllers to the PC.
// Wire formats: src/telescene/wire.py. Log tag: telescene.
#include <android/log.h>
#include <android/sensor.h>
#include <android_native_app_glue.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <jni.h>
#include <math.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/system_properties.h>
#include <time.h>
#include <unistd.h>
#define XR_USE_PLATFORM_ANDROID
#define XR_USE_GRAPHICS_API_OPENGL_ES
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#define LOG(...) __android_log_print(ANDROID_LOG_INFO, "telescene", __VA_ARGS__)
#define CHECK(call) do { XrResult r_ = (call); if (XR_FAILED(r_)) LOG("FAILED %s = %d", #call, r_); } while (0)
enum { SCENE_PORT = 47800, POSE_PORT = 47801, INPUT_PORT = 47802, BEACON_PORT = 47803, MAX_POSES = 512, JOINTS = 24 };

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// ---------------------------------------------------------------- math
typedef struct { float m[16]; } Mat;  // column major
static Mat mul(Mat a, Mat b) {
    Mat r;
    for (int c = 0; c < 4; c++) for (int i = 0; i < 4; i++) {
        float s = 0; for (int k = 0; k < 4; k++) s += a.m[k * 4 + i] * b.m[c * 4 + k];
        r.m[c * 4 + i] = s;
    }
    return r;
}
static Mat from_pose(const float* p, const float* q) {  // p xyz, q xyzw
    float x = q[0], y = q[1], z = q[2], w = q[3];
    return (Mat){{1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0,
                  2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
                  2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0,
                  p[0], p[1], p[2], 1}};
}
static Mat rigid_inverse(Mat a) {
    Mat r = {{a.m[0], a.m[4], a.m[8], 0, a.m[1], a.m[5], a.m[9], 0, a.m[2], a.m[6], a.m[10], 0, 0, 0, 0, 1}};
    for (int i = 0; i < 3; i++) r.m[12 + i] = -(r.m[i] * a.m[12] + r.m[4 + i] * a.m[13] + r.m[8 + i] * a.m[14]);
    return r;
}
static Mat projection(XrFovf f, float zn, float zf) {
    float l = tanf(f.angleLeft), r = tanf(f.angleRight), d = tanf(f.angleDown), u = tanf(f.angleUp);
    return (Mat){{2 / (r - l), 0, 0, 0, 0, 2 / (u - d), 0, 0, (r + l) / (r - l), (u + d) / (u - d), -(zf + zn) / (zf - zn), -1,
                  0, 0, -2 * zf * zn / (zf - zn), 0}};
}
// Simulator (Z up, metres) to OpenXR local floor (Y up, -Z forward): the anchor's forward becomes -Z.
static Mat sim_to_xr(const float* a, float yaw, float scale) {
    float c = cosf(yaw), s = sinf(yaw), k = 1.0f / scale;
    // x' = c dx + s dy, y' = -s dx + c dy, then xr = k * (-y', dz, -x')
    Mat m = {{s * k, 0, -c * k, 0, -c * k, 0, -s * k, 0, 0, k, 0, 0, 0, 0, 0, 1}};
    m.m[12] = -(m.m[0] * a[0] + m.m[4] * a[1]);
    m.m[13] = -(m.m[9] * a[2]);
    m.m[14] = -(m.m[2] * a[0] + m.m[6] * a[1]);
    return m;
}

// ---------------------------------------------------------------- scene download
typedef struct { int pose; float rgb[3]; GLuint vao, vbo, ibo; GLsizei count; } Draw;
static Draw* g_draws; static int g_num_draws;
static uint8_t* _Atomic g_blob_ready; static size_t g_blob_size;
enum { MODE_MENU, MODE_CONNECT, MODE_MIRROR, MODE_SCENES };
static atomic_int g_mode, g_scene_fd = -1;
// Scenes the PC offers (request 'L'), and the one the operator picked (request 'C'). The PC restarts with it.
#define MAX_SCENES 96  // telescene.wire.MAX_SCENES
#define SCENE_NAME 48
static pthread_mutex_t g_scene_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_scene_names[MAX_SCENES][SCENE_NAME]; static int g_num_scenes, g_scene_current = -1; static bool g_scene_list_tried;
static atomic_int g_choice = -1; static atomic_bool g_loading; static char g_loading_name[SCENE_NAME];
// While the PC starts: how far it has got (PROGRESS_PORT), and what the bar shows, which only moves forward.
static float g_progress, g_progress_shown; static char g_progress_text[SCENE_NAME];
static char g_pc_ip[64];
static atomic_bool g_quit;

static bool read_all(int fd, void* out, size_t n) {
    uint8_t* p = out;
    while (n) { ssize_t k = recv(fd, p, n, 0); if (k <= 0) return false; p += k; n -= k; }
    return true;
}
static int connect_port(int port, double timeout_s) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)};
    inet_pton(AF_INET, g_pc_ip, &a.sin_addr);
    if (timeout_s > 0) {  // also bounds connect() itself
        struct timeval to = {(time_t)timeout_s, (suseconds_t)((timeout_s - (int)timeout_s) * 1e6)};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof to); setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &to, sizeof to);
    }
    if (connect(fd, (struct sockaddr*)&a, sizeof a) != 0) { close(fd); return -1; }
    return fd;
}
static int connect_pc(double timeout_s) { return connect_port(SCENE_PORT, timeout_s); }
// "XPRG", fraction (f32), then a length byte and the stage's name.
static void fetch_progress(void) {
    int fd = connect_port(SCENE_PORT + 4, 0.5); if (fd < 0) return;
    uint8_t h[9]; char text[SCENE_NAME];
    if (read_all(fd, h, sizeof h) && !memcmp(h, "XPRG", 4) && h[8] < SCENE_NAME && read_all(fd, text, h[8])) {
        text[h[8]] = 0; float f; memcpy(&f, h + 4, 4);
        pthread_mutex_lock(&g_scene_lock); g_progress = f; memcpy(g_progress_text, text, sizeof text); pthread_mutex_unlock(&g_scene_lock);
    }
    close(fd);
}
// "XLST", count, current, then per scene a length byte and that many ASCII bytes.
static void fetch_scene_list(void) {
    int fd = connect_pc(1.0); if (fd < 0) return;
    char request = 'L'; send(fd, &request, 1, MSG_NOSIGNAL);
    uint32_t h[3]; char names[MAX_SCENES][SCENE_NAME]; bool ok = false;
    if (read_all(fd, h, sizeof h) && !memcmp(h, "XLST", 4) && h[1] <= MAX_SCENES) {
        ok = true;
        for (uint32_t i = 0; i < h[1] && ok; i++) {
            uint8_t n = 0; ok = read_all(fd, &n, 1) && n < SCENE_NAME && read_all(fd, names[i], n);
            if (ok) names[i][n] = 0;
        }
    }
    close(fd);
    pthread_mutex_lock(&g_scene_lock);
    if (ok) { memcpy(g_scene_names, names, sizeof names); g_num_scenes = (int)h[1]; g_scene_current = (int)h[2]; }
    g_scene_list_tried = true;
    pthread_mutex_unlock(&g_scene_lock);
}
static void send_scene_choice(int choice) {
    int fd = connect_pc(2.0); if (fd < 0) { LOG("could not reach the PC to change scene"); return; }
    uint8_t request[2] = {'C', (uint8_t)choice}; send(fd, request, 2, MSG_NOSIGNAL);
    char ok[2] = {0}; read_all(fd, ok, 2); close(fd);
    LOG("asked the PC for scene %d: %s", choice, ok[0] == 'O' ? "restarting" : "no answer");
}
static void* net_thread(void* arg) {
    (void)arg; double last_list = 0;
    while (!atomic_load(&g_quit)) {
        int mode = atomic_load(&g_mode);
        if (mode == MODE_MENU || mode == MODE_SCENES) {
            if (mode == MODE_SCENES && now_s() - last_list > 2.0) { fetch_scene_list(); last_list = now_s(); }
            if (mode == MODE_MENU) last_list = 0;
            usleep(100000); continue;
        }
        last_list = 0;  // back on the menu, ask for the list at once
        int choice = atomic_exchange(&g_choice, -1);
        if (choice >= 0) send_scene_choice(choice);
        int fd = connect_pc(0);
        if (fd < 0) {
            if (atomic_load(&g_loading)) { fetch_progress(); usleep(400000); } else sleep(1);
            continue;
        }
        char request = mode == MODE_MIRROR ? 'M' : 'S'; send(fd, &request, 1, MSG_NOSIGNAL);
        atomic_store(&g_scene_fd, fd);
        LOG("connected to %s:%d, asked for scene %c", g_pc_ip, SCENE_PORT, request);
        uint32_t h[4];
        if (read_all(fd, h, sizeof h) && !memcmp(h, "XSCN", 4)) {
            size_t cap = 1 << 24, used = sizeof h; uint8_t* b = malloc(cap); memcpy(b, h, sizeof h);
            uint32_t nodes = h[3]; bool ok = true;
            for (uint32_t i = 0; i < nodes && ok; i++) {
                uint8_t nh[24];
                if (!(ok = read_all(fd, nh, 24))) break;
                uint32_t nv, ni; memcpy(&nv, nh + 16, 4); memcpy(&ni, nh + 20, 4);
                size_t need = 24 + (size_t)nv * 12 + (size_t)ni * 4;
                if (used + need > cap) { while (used + need > cap) cap *= 2; b = realloc(b, cap); }
                memcpy(b + used, nh, 24);
                ok = read_all(fd, b + used + 24, need - 24);
                used += need;
            }
            if (ok) { g_blob_size = used; LOG("scene received: %u nodes, %zu bytes", nodes, used); atomic_store(&g_loading, false);
                free(atomic_exchange(&g_blob_ready, b)); } else { free(b); LOG("scene download failed"); }
            char c; while (!atomic_load(&g_quit) && recv(fd, &c, 1, 0) > 0) {}  // held open: EOF means PC restarted
        }
        atomic_store(&g_scene_fd, -1);
        close(fd);
        if (atomic_load(&g_mode) == mode) sleep(1);  // lost the PC. After a mode change, go straight away.
    }
    return NULL;
}

// ---------------------------------------------------------------- GL
static GLuint g_prog, g_cube_vao; static GLint u_mvp, u_mv, u_color;
static const char* VS = "#version 300 es\nlayout(location=0) in vec3 p; uniform mat4 mvp; uniform mat4 mv; out vec3 v;"
    "void main(){v=(mv*vec4(p,1.)).xyz; gl_Position=mvp*vec4(p,1.);}";
// Flat shading from screen derivatives: no normals in the scene blob and no extra vertices for them.
static const char* FS = "#version 300 es\nprecision mediump float; in vec3 v; uniform vec4 color; out vec4 o;"
    "void main(){vec3 n=normalize(cross(dFdx(v),dFdy(v))); float d=abs(dot(n,normalize(vec3(.3,.8,.5))));"
    "float f=abs(n.y);o=vec4(color.rgb*(.25+.55*d+.2*f),1.);}";
static GLuint compile(GLenum t, const char* s) {
    GLuint h = glCreateShader(t); glShaderSource(h, 1, &s, NULL); glCompileShader(h);
    GLint ok; glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) { char l[512]; glGetShaderInfoLog(h, 512, NULL, l); LOG("shader: %s", l); }
    return h;
}
static void gl_init(void) {
    g_prog = glCreateProgram();
    glAttachShader(g_prog, compile(GL_VERTEX_SHADER, VS)); glAttachShader(g_prog, compile(GL_FRAGMENT_SHADER, FS));
    glLinkProgram(g_prog);
    u_mvp = glGetUniformLocation(g_prog, "mvp"); u_mv = glGetUniformLocation(g_prog, "mv"); u_color = glGetUniformLocation(g_prog, "color");
    static const float cv[] = {-1,-1,-1, 1,-1,-1, 1,1,-1, -1,1,-1, -1,-1,1, 1,-1,1, 1,1,1, -1,1,1};
    static const uint16_t ci[] = {0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,6,2,3,7,6,0,7,3,0,4,7,1,2,6,1,6,5};
    GLuint b[2]; glGenVertexArrays(1, &g_cube_vao); glBindVertexArray(g_cube_vao); glGenBuffers(2, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]); glBufferData(GL_ARRAY_BUFFER, sizeof cv, cv, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b[1]); glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof ci, ci, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindVertexArray(0);
}
static void upload_scene(uint8_t* b) {
    for (int i = 0; i < g_num_draws; i++) { glDeleteVertexArrays(1, &g_draws[i].vao); glDeleteBuffers(1, &g_draws[i].vbo); glDeleteBuffers(1, &g_draws[i].ibo); }
    uint32_t h[4]; memcpy(h, b, 16);
    free(g_draws); g_draws = calloc(h[3], sizeof(Draw)); g_num_draws = h[3];
    size_t o = 16; long tris = 0;
    for (uint32_t i = 0; i < h[3]; i++) {
        Draw* d = &g_draws[i]; uint32_t nv, ni;
        memcpy(&d->pose, b + o, 4); memcpy(d->rgb, b + o + 4, 12); memcpy(&nv, b + o + 16, 4); memcpy(&ni, b + o + 20, 4); o += 24;
        glGenVertexArrays(1, &d->vao); glBindVertexArray(d->vao); glGenBuffers(1, &d->vbo); glGenBuffers(1, &d->ibo);
        glBindBuffer(GL_ARRAY_BUFFER, d->vbo); glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)nv * 12, b + o, GL_STATIC_DRAW); o += (size_t)nv * 12;
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, d->ibo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)ni * 4, b + o, GL_STATIC_DRAW); o += (size_t)ni * 4;
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        d->count = ni; tris += ni / 3;
    }
    glBindVertexArray(0);
    LOG("scene uploaded: %d draws, %ld triangles", g_num_draws, tris);
}

// ---------------------------------------------------------------- pose stream with interpolation
static float g_prev[MAX_POSES][7], g_cur[MAX_POSES][7], g_anchor[3], g_yaw, g_scale = 1.0f, g_haptic[2]; static uint32_t g_status;
static uint32_t g_num_poses; static double g_cur_t, g_period = 0.02; static long g_pose_packets; static bool g_have_pose;
static int g_pose_fd = -1, g_input_fd = -1; static struct sockaddr_in g_pc;
static bool g_input_udp; static atomic_bool g_pc_known;
// The render thread only copies the newest input here. A separate thread does the sending, so a stalled
// link can't slow the display, and frames that pile up meanwhile are replaced rather than queued.
#define INPUT_BYTES (4 + 4 + 4 + (8 + 7 + JOINTS * 7 + 2 * 7) * 4)
static pthread_mutex_t g_in_lock = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_in_cond = PTHREAD_COND_INITIALIZER;
static uint8_t g_in_box[INPUT_BYTES]; static bool g_in_full;
static bool send_all(int fd, const uint8_t* p, size_t n) {
    while (n) { ssize_t k = send(fd, p, n, MSG_NOSIGNAL); if (k <= 0) return false; p += k; n -= k; }
    return true;
}
static void* input_thread(void* arg) {
    (void)arg; int fd = -1; uint8_t frame[INPUT_BYTES];
    while (!atomic_load(&g_quit)) {
        pthread_mutex_lock(&g_in_lock);
        while (!g_in_full) pthread_cond_wait(&g_in_cond, &g_in_lock);
        memcpy(frame, g_in_box, sizeof frame); g_in_full = false;
        pthread_mutex_unlock(&g_in_lock);
        if (fd < 0) {
            fd = socket(AF_INET, SOCK_STREAM, 0);
            int one = 1, buf = 4096, user_ms = 2000; struct timeval to = {2, 0};
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);  // stale frames must not queue in the kernel
            setsockopt(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, &user_ms, sizeof user_ms);
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &to, sizeof to);
            if (connect(fd, (struct sockaddr*)&g_pc, sizeof g_pc) != 0) { close(fd); fd = -1; usleep(200000); continue; }
            LOG("input connected over TCP to %s:%d", g_pc_ip, INPUT_PORT);
        }
        if (!send_all(fd, frame, sizeof frame)) { LOG("input link lost: %d", errno); close(fd); fd = -1; }
    }
    return NULL;
}

static void net_udp_init(void) {
    g_pose_fd = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(POSE_PORT), .sin_addr.s_addr = INADDR_ANY};
    if (bind(g_pose_fd, (struct sockaddr*)&a, sizeof a)) LOG("pose bind failed: %d", errno);
    fcntl(g_pose_fd, F_SETFL, O_NONBLOCK);
    g_input_fd = socket(AF_INET, SOCK_DGRAM, 0); fcntl(g_input_fd, F_SETFL, O_NONBLOCK);
    g_pc = (struct sockaddr_in){.sin_family = AF_INET, .sin_port = htons(INPUT_PORT)};
    inet_pton(AF_INET, g_pc_ip, &g_pc.sin_addr);
}
// With no address set, wait for the PC's broadcast. Nothing else on the network side starts before this ends.
static void* start_network(void* arg) {
    (void)arg;
    if (!g_pc_ip[0]) {
        int fd = socket(AF_INET, SOCK_DGRAM, 0), one = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(BEACON_PORT), .sin_addr.s_addr = INADDR_ANY};
        if (bind(fd, (struct sockaddr*)&a, sizeof a)) LOG("beacon bind failed: %d", errno);
        LOG("no address set, waiting for the PC's broadcast");
        char m[8]; struct sockaddr_in from; socklen_t len;
        while (!atomic_load(&g_quit)) {
            len = sizeof from; ssize_t n = recvfrom(fd, m, sizeof m, 0, (struct sockaddr*)&from, &len);
            if (n == 4 && !memcmp(m, "XBCN", 4)) { inet_ntop(AF_INET, &from.sin_addr, g_pc_ip, sizeof g_pc_ip); break; }
        }
        close(fd);
        LOG("found the PC at %s", g_pc_ip);
    }
    inet_pton(AF_INET, g_pc_ip, &g_pc.sin_addr);
    atomic_store(&g_pc_known, true);
    pthread_t ti; if (!g_input_udp) pthread_create(&ti, NULL, input_thread, NULL);
    return net_thread(NULL);
}

static void poll_poses(void) {
    static uint8_t buf[16384]; ssize_t n;
    while ((n = recv(g_pose_fd, buf, sizeof buf, 0)) >= 44) {
        uint32_t hdr[3]; memcpy(hdr, buf, 12);
        if (memcmp(buf, "XPOS", 4) || hdr[2] > MAX_POSES || (size_t)n < 44 + (size_t)hdr[2] * 28) continue;
        memcpy(g_prev, g_cur, sizeof g_prev);
        memcpy(g_anchor, buf + 12, 12); memcpy(&g_yaw, buf + 24, 4); memcpy(&g_scale, buf + 28, 4); memcpy(g_haptic, buf + 32, 8); memcpy(&g_status, buf + 40, 4);
        memcpy(g_cur, buf + 44, hdr[2] * 28); g_num_poses = hdr[2];
        double t = now_s();
        if (g_have_pose) { double d = t - g_cur_t; g_period += 0.05 * ((d < 0.2 ? d : g_period) - g_period); }
        else memcpy(g_prev, g_cur, sizeof g_prev);
        g_cur_t = t; g_have_pose = true; g_pose_packets++;
    }
}
// Display runs faster than the stream. Blending prev -> cur over one period gives smooth motion at the
// cost of one stream period of latency on the robot only: the operator's own head is not affected.
static Mat link_matrix(int i, float alpha) {
    const float* a = g_prev[i]; const float* b = g_cur[i];
    float p[3], q[4], dot = 0, sign;
    for (int k = 0; k < 4; k++) dot += a[3 + k] * b[3 + k];
    sign = dot < 0 ? -1.f : 1.f;
    for (int k = 0; k < 3; k++) p[k] = a[k] + (b[k] - a[k]) * alpha;
    float l = 0;
    for (int k = 0; k < 4; k++) { q[k] = a[3 + k] + (sign * b[3 + k] - a[3 + k]) * alpha; l += q[k] * q[k]; }
    l = l > 1e-8f ? 1.f / sqrtf(l) : 0.f;
    for (int k = 0; k < 4; k++) q[k] *= l;
    return from_pose(p, q);
}

// ---------------------------------------------------------------- OpenXR
typedef struct { XrSwapchain sc; uint32_t n; XrSwapchainImageOpenGLESKHR* img; GLuint* fbo; GLuint depth; int w, h; } Eye;
static XrInstance g_inst; static XrSession g_sess; static XrSpace g_space, g_view_space; static Eye g_eye[2];
static XrSessionState g_state = XR_SESSION_STATE_UNKNOWN; static bool g_running;
static XrActionSet g_set; static XrAction a_trig, a_grip, a_stick, a_abtn, a_bbtn, a_xbtn, a_ybtn, a_haptic, a_grip_pose, a_aim_pose; static XrSpace g_grip_space[2], g_aim_space[2]; static XrPath g_hand[2];
static XrBodyTrackerBD g_body; static PFN_xrLocateBodyJointsBD pfnLocate;
static PFN_xrCreatePassthroughFB pfnPtCreate; static PFN_xrCreatePassthroughLayerFB pfnPtLayer;
static PFN_xrPassthroughStartFB pfnPtStart; static XrPassthroughFB g_pt; static XrPassthroughLayerFB g_pt_layer;
static bool g_has_passthrough; static float g_body_out[JOINTS * 7]; static uint32_t g_flags;
static long g_frames, g_body_frames; static int g_body_valid;
static void (*pfnFbMs)(GLenum, GLenum, GLenum, GLuint, GLint, GLsizei);
static void (*pfnRbMs)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);

static XrPath path(const char* s) { XrPath p; xrStringToPath(g_inst, s, &p); return p; }
static XrAction make_action(const char* name, XrActionType t, bool both) {
    XrActionCreateInfo i = {XR_TYPE_ACTION_CREATE_INFO}; strcpy(i.actionName, name); strcpy(i.localizedActionName, name);
    i.actionType = t; i.countSubactionPaths = both ? 2 : 0; i.subactionPaths = both ? g_hand : NULL;
    XrAction a; CHECK(xrCreateAction(g_set, &i, &a)); return a;
}
static void actions_init(void) {
    XrActionSetCreateInfo si = {XR_TYPE_ACTION_SET_CREATE_INFO}; strcpy(si.actionSetName, "teleop"); strcpy(si.localizedActionSetName, "teleop");
    CHECK(xrCreateActionSet(g_inst, &si, &g_set));
    g_hand[0] = path("/user/hand/left"); g_hand[1] = path("/user/hand/right");
    a_trig = make_action("trigger", XR_ACTION_TYPE_FLOAT_INPUT, true);
    a_grip = make_action("grip", XR_ACTION_TYPE_FLOAT_INPUT, true);
    a_stick = make_action("stick", XR_ACTION_TYPE_VECTOR2F_INPUT, true);
    a_abtn = make_action("a_button", XR_ACTION_TYPE_BOOLEAN_INPUT, false);
    a_bbtn = make_action("b_button", XR_ACTION_TYPE_BOOLEAN_INPUT, false);
    a_xbtn = make_action("x_button", XR_ACTION_TYPE_BOOLEAN_INPUT, false);
    a_ybtn = make_action("y_button", XR_ACTION_TYPE_BOOLEAN_INPUT, false);
    a_grip_pose = make_action("grip_pose", XR_ACTION_TYPE_POSE_INPUT, true);
    a_aim_pose = make_action("aim_pose", XR_ACTION_TYPE_POSE_INPUT, true);
    a_haptic = make_action("haptic", XR_ACTION_TYPE_VIBRATION_OUTPUT, true);
    XrActionSuggestedBinding b[] = {
        {a_trig, path("/user/hand/left/input/trigger/value")}, {a_trig, path("/user/hand/right/input/trigger/value")},
        {a_grip, path("/user/hand/left/input/squeeze/value")}, {a_grip, path("/user/hand/right/input/squeeze/value")},
        {a_stick, path("/user/hand/left/input/thumbstick")}, {a_stick, path("/user/hand/right/input/thumbstick")},
        {a_abtn, path("/user/hand/right/input/a/click")}, {a_bbtn, path("/user/hand/right/input/b/click")},
        {a_xbtn, path("/user/hand/left/input/x/click")}, {a_ybtn, path("/user/hand/left/input/y/click")},
        {a_grip_pose, path("/user/hand/left/input/grip/pose")}, {a_grip_pose, path("/user/hand/right/input/grip/pose")},
        {a_aim_pose, path("/user/hand/left/input/aim/pose")}, {a_aim_pose, path("/user/hand/right/input/aim/pose")},
        {a_haptic, path("/user/hand/left/output/haptic")}, {a_haptic, path("/user/hand/right/output/haptic")}};
    XrInteractionProfileSuggestedBinding s = {XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    s.suggestedBindings = b; s.countSuggestedBindings = sizeof b / sizeof b[0];
    const char* ps[] = {"/interaction_profiles/bytedance/pico4_controller", "/interaction_profiles/bytedance/pico4s_controller"};
    for (int i = 0; i < 2; i++) { s.interactionProfile = path(ps[i]); XrResult r = xrSuggestInteractionProfileBindings(g_inst, &s); LOG("bindings %s = %d", ps[i], r); }
}
static void passthrough_init(void) {
    if (!g_has_passthrough) return;
    xrGetInstanceProcAddr(g_inst, "xrCreatePassthroughFB", (PFN_xrVoidFunction*)&pfnPtCreate);
    xrGetInstanceProcAddr(g_inst, "xrCreatePassthroughLayerFB", (PFN_xrVoidFunction*)&pfnPtLayer);
    xrGetInstanceProcAddr(g_inst, "xrPassthroughStartFB", (PFN_xrVoidFunction*)&pfnPtStart);
    PFN_xrPassthroughLayerResumeFB resume = NULL;
    xrGetInstanceProcAddr(g_inst, "xrPassthroughLayerResumeFB", (PFN_xrVoidFunction*)&resume);
    XrPassthroughCreateInfoFB c = {XR_TYPE_PASSTHROUGH_CREATE_INFO_FB};
    XrResult r = pfnPtCreate(g_sess, &c, &g_pt); LOG("passthrough create = %d", r);
    if (XR_FAILED(r)) { g_has_passthrough = false; return; }
    XrPassthroughLayerCreateInfoFB l = {XR_TYPE_PASSTHROUGH_LAYER_CREATE_INFO_FB};
    l.passthrough = g_pt; l.purpose = XR_PASSTHROUGH_LAYER_PURPOSE_RECONSTRUCTION_FB;
    r = pfnPtLayer(g_sess, &l, &g_pt_layer); LOG("passthrough layer = %d", r);
    if (XR_FAILED(r)) { g_has_passthrough = false; return; }
    pfnPtStart(g_pt); if (resume) resume(g_pt_layer);
}
static void body_init(void) {
    PFN_xrCreateBodyTrackerBD create = NULL;
    xrGetInstanceProcAddr(g_inst, "xrCreateBodyTrackerBD", (PFN_xrVoidFunction*)&create);
    xrGetInstanceProcAddr(g_inst, "xrLocateBodyJointsBD", (PFN_xrVoidFunction*)&pfnLocate);
    if (!create || !pfnLocate) { LOG("RESULT no body tracking entry points"); return; }
    XrBodyTrackerCreateInfoBD c = {XR_TYPE_BODY_TRACKER_CREATE_INFO_BD}; c.jointSet = XR_BODY_JOINT_SET_FULL_BODY_JOINTS_BD;
    XrResult r = create(g_sess, &c, &g_body); LOG("RESULT xrCreateBodyTrackerBD = %d", r);
    if (XR_FAILED(r)) g_body = XR_NULL_HANDLE;
}
static void read_body(XrTime t) {
    g_flags &= ~6u; g_body_valid = 0; memset(g_body_out, 0, sizeof g_body_out);
    if (!g_body) return;
    XrBodyJointLocationBD j[JOINTS];
    XrBodyJointsLocateInfoBD i = {XR_TYPE_BODY_JOINTS_LOCATE_INFO_BD}; i.baseSpace = g_space; i.time = t;
    XrBodyJointLocationsBD l = {XR_TYPE_BODY_JOINT_LOCATIONS_BD}; l.jointLocationCount = JOINTS; l.jointLocations = j;
    XrResult r = pfnLocate(g_body, &i, &l);
    if (XR_FAILED(r)) { static int once; if (!once++) LOG("xrLocateBodyJointsBD = %d", r); return; }
    g_flags |= 2u; if (l.allJointPosesTracked) g_flags |= 4u;
    for (int k = 0; k < JOINTS; k++) if (j[k].locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) {
        memcpy(&g_body_out[k * 7], &j[k].pose.position, 12); memcpy(&g_body_out[k * 7 + 3], &j[k].pose.orientation, 16); g_body_valid++;
    }
    g_body_frames++;
}
static float get_float(XrAction a, XrPath sub) {
    XrActionStateGetInfo i = {XR_TYPE_ACTION_STATE_GET_INFO}; i.action = a; i.subactionPath = sub;
    XrActionStateFloat s = {XR_TYPE_ACTION_STATE_FLOAT}; return XR_SUCCEEDED(xrGetActionStateFloat(g_sess, &i, &s)) && s.isActive ? s.currentState : 0;
}
// The proximity sensor reads 0 on a face and 5 otherwise. It tells the PC the headset came off, so that the
// robot does not copy the posture of a headset on a desk. Short blips are ignored; no sensor means worn.
#define PROX_LOOPER_ID (LOOPER_ID_USER + 1)
static ASensorEventQueue* g_prox_queue; static bool g_prox_far; static double g_prox_far_since;
static void prox_init(void) {
    ASensorManager* m = ASensorManager_getInstanceForPackage("org.telescene.app");
    const ASensor* s = ASensorManager_getDefaultSensor(m, ASENSOR_TYPE_PROXIMITY);
    if (!s) { LOG("no proximity sensor"); return; }
    g_prox_queue = ASensorManager_createEventQueue(m, ALooper_forThread(), PROX_LOOPER_ID, NULL, NULL);
    ASensorEventQueue_enableSensor(g_prox_queue, s);
}
static void prox_drain(void) {
    ASensorEvent e;
    while (ASensorEventQueue_getEvents(g_prox_queue, &e, 1) > 0) {
        bool far = e.distance >= 1.0f;
        if (far && !g_prox_far) g_prox_far_since = now_s();
        g_prox_far = far; LOG("proximity %.1f", e.distance);
    }
}
// Each pulse is slightly longer than the gap to the next, so that a steady contact feels continuous. If poses stop
// arriving the pulses stop too: nothing keeps buzzing after the PC has gone.
static void play_haptics(void) {
    static double next; double t = now_s();
    if (t < next || t - g_cur_t > 0.2) return;
    next = t + 0.05;
    for (int h = 0; h < 2; h++) {
        if (g_haptic[h] <= 0) continue;
        XrHapticActionInfo info = {XR_TYPE_HAPTIC_ACTION_INFO}; info.action = a_haptic; info.subactionPath = g_hand[h];
        XrHapticVibration v = {XR_TYPE_HAPTIC_VIBRATION};
        v.amplitude = g_haptic[h]; v.duration = 70000000; v.frequency = XR_FREQUENCY_UNSPECIFIED;
        XrResult r = xrApplyHapticFeedback(g_sess, &info, (const XrHapticBaseHeader*)&v);
        static XrResult logged = 1; if (r != logged) { logged = r; LOG("haptic result %d", r); }
    }
}
// The menu: two panels in the room, in front of where the head was when it opened. Point a controller at one and
// pull its trigger (or press A). X comes back here from either mode, and on the menu brings it in front again.
static float g_head_p[3], g_head_q[4] = {0, 0, 0, 1}; static bool g_head_valid;
static float g_aim_p[2][3], g_aim_q[2][4]; static bool g_aim_valid[2]; static float g_ray_len[2];
static float g_menu_at[3], g_menu_yaw; static bool g_menu_placed; static int g_menu_look = -1, g_menu_hand = -1;
// A panel of the open menu: centre and half sizes in the menu's plane, what it does, and its text.
enum { P_MAIN, P_SCENE, P_TAB, P_PREV, P_NEXT, P_MIRROR, P_CONNECT };
typedef struct { float x, y, w, h; int kind, idx; char text[SCENE_NAME]; } Panel;
static const float RAY_MAX = 2.0f;
enum { MAIN_ITEMS = 3 };
static const char* MAIN_LABEL[MAIN_ITEMS] = {"CONNECT", "MIRROR", "SCENE"};
// The scene menu: a tab per group (the PC sends "group\x1fname"), then a page of scenes in two columns.
#define MAX_TABS 8
#define PAGE_ROWS 6
#define MENU_PAGE (2 * PAGE_ROWS)
#define MAX_PANELS (MAX_TABS + MENU_PAGE + 4)
static int g_scene_tab, g_scene_page, g_scene_pages = 1; static bool g_scene_view_set;
static void split_scene(const char* raw, char* group, char* label) {
    const char* sep = strchr(raw, 0x1f);
    if (!sep) { group[0] = 0; snprintf(label, SCENE_NAME, "%s", raw); return; }
    snprintf(group, SCENE_NAME, "%.*s", (int)(sep - raw), raw); snprintf(label, SCENE_NAME, "%s", sep + 1);
}
// The panels of the open menu: three across for the main menu; tabs, a page of scenes and PREV / NEXT for the
// scene menu. It opens on the running scene's tab and page.
static int menu_panels(Panel* out) {
    if (atomic_load(&g_mode) != MODE_SCENES) {
        for (int i = 0; i < MAIN_ITEMS; i++) {
            out[i] = (Panel){(i - 1) * 0.5f, 0, 0.23f, 0.12f, P_MAIN, i, ""};
            snprintf(out[i].text, SCENE_NAME, "%s", MAIN_LABEL[i]);
        }
        return MAIN_ITEMS;
    }
    static char names[MAX_SCENES][SCENE_NAME], groups[MAX_SCENES][SCENE_NAME], labels[MAX_SCENES][SCENE_NAME];
    pthread_mutex_lock(&g_scene_lock);
    int n = g_num_scenes, current = g_scene_current; memcpy(names, g_scene_names, sizeof names);
    pthread_mutex_unlock(&g_scene_lock);
    char tabs[MAX_TABS][SCENE_NAME]; int n_tabs = 0;
    for (int i = 0; i < n; i++) {
        split_scene(names[i], groups[i], labels[i]);
        int t = 0; while (t < n_tabs && t < MAX_TABS && strcmp(tabs[t], groups[i])) t++;
        if (t == n_tabs) { if (n_tabs < MAX_TABS) snprintf(tabs[n_tabs], SCENE_NAME, "%s", groups[i]); n_tabs++; }
    }
    if (n_tabs < 2 || n_tabs > MAX_TABS) n_tabs = 0;  // one group, or too many for a row of tabs: one list
    if (!g_scene_view_set && n > 0) {
        g_scene_tab = g_scene_page = 0;
        if (current >= 0 && current < n) {
            int pos = 0;
            for (int t = 0; t < n_tabs; t++) if (!strcmp(tabs[t], groups[current])) g_scene_tab = t;
            for (int i = 0; i < current; i++) if (!n_tabs || !strcmp(groups[i], groups[current])) pos++;
            g_scene_page = pos / MENU_PAGE;
        }
        g_scene_view_set = true;
    }
    if (g_scene_tab >= (n_tabs ? n_tabs : 1)) g_scene_tab = 0;
    int members[MAX_SCENES], m = 0;
    for (int i = 0; i < n; i++) if (!n_tabs || !strcmp(groups[i], tabs[g_scene_tab])) members[m++] = i;
    g_scene_pages = m ? (m + MENU_PAGE - 1) / MENU_PAGE : 1;
    if (g_scene_page >= g_scene_pages) g_scene_page = g_scene_pages - 1;
    if (g_scene_page < 0) g_scene_page = 0;
    int k = 0;
    float width = 1.3f / (n_tabs ? n_tabs : 1);
    for (int t = 0; t < n_tabs; t++) {
        out[k] = (Panel){-0.65f + width * (t + 0.5f), 0.36f, width / 2 - 0.01f, 0.035f, P_TAB, t, ""};
        snprintf(out[k++].text, SCENE_NAME, "%s", tabs[t]);
    }
    int first = g_scene_page * MENU_PAGE, on_page = m - first < MENU_PAGE ? m - first : MENU_PAGE;
    bool one_column = g_scene_pages == 1 && on_page <= PAGE_ROWS;
    float top = n_tabs ? 0.25f : 0.34f, step = 0.09f, bottom = top;
    for (int j = 0; j < on_page; j++) {
        int row = one_column ? j : j % PAGE_ROWS, col = one_column ? 0 : j / PAGE_ROWS;
        float x = one_column ? 0 : (col ? 0.33f : -0.33f), y = top - row * step;
        out[k] = (Panel){x, y, one_column ? 0.42f : 0.31f, 0.035f, P_SCENE, members[first + j], ""};
        snprintf(out[k++].text, SCENE_NAME, "%s", labels[members[first + j]]);
        if (y < bottom) bottom = y;
    }
    if (!m && g_scene_list_tried) {  // a PC without a scene list: just join what it runs
        out[k++] = (Panel){0, top, 0.42f, 0.05f, P_CONNECT, 0, "CONNECT"};
        bottom = top;
    }
    if (g_scene_pages > 1) {
        bottom -= step * 1.1f;
        out[k++] = (Panel){-0.33f, bottom, 0.12f, 0.035f, P_PREV, 0, "< PREV"};
        out[k++] = (Panel){0.33f, bottom, 0.12f, 0.035f, P_NEXT, 0, "NEXT >"};
    }
    // Not a scene: see your own body. Set apart below the list, in its own colour.
    out[k++] = (Panel){0, bottom - step * 1.6f, 0.24f, 0.05f, P_MIRROR, 0, "MIRROR"};
    return k;
}
static void set_mode(int mode) {
    atomic_store(&g_mode, mode); g_have_pose = false; g_menu_placed = false; g_menu_look = -1; g_menu_hand = -1;
    g_scene_view_set = false;
    int fd = atomic_load(&g_scene_fd); if (fd >= 0) shutdown(fd, SHUT_RDWR);
    LOG("mode %d", mode);
}
static void forward_of(const float* q, float* d) {  // -Z of an OpenXR pose
    d[0] = -2 * (q[0] * q[2] + q[3] * q[1]); d[1] = -2 * (q[1] * q[2] - q[3] * q[0]); d[2] = -(1 - 2 * (q[0] * q[0] + q[1] * q[1]));
}
// The panel a ray from p along d hits, or -1, and how far along the ray.
static int menu_hit(const float* p, const float* d, float* along, const Panel* panels, int n) {
    float c = cosf(g_menu_yaw), s = sinf(g_menu_yaw);
    float o[3] = {p[0] - g_menu_at[0], p[1] - g_menu_at[1], p[2] - g_menu_at[2]};
    float ox = c * o[0] - s * o[2], oz = s * o[0] + c * o[2], dx = c * d[0] - s * d[2], dz = s * d[0] + c * d[2];
    *along = RAY_MAX;
    if (dz > -1e-3f) return -1;
    float k = -oz / dz; if (k > RAY_MAX) return -1;
    float x = ox + k * dx, y = o[1] + k * d[1];
    for (int i = 0; i < n; i++) if (fabsf(x - panels[i].x) < panels[i].w && fabsf(y - panels[i].y) < panels[i].h) { *along = k; return i; }
    *along = k;  // on the menu's plane but between panels: the beam still stops there
    return -1;
}
static void menu_input(const bool trigger[2], bool a_button, bool back) {
    static bool held[2] = {true, true}, held_a = true, held_back;
    bool pulled[2] = {trigger[0] && !held[0], trigger[1] && !held[1]}, a_now = a_button && !held_a, back_now = back && !held_back;
    held[0] = trigger[0]; held[1] = trigger[1]; held_a = a_button; held_back = back;
    int mode = atomic_load(&g_mode);
    if (mode != MODE_MENU && mode != MODE_SCENES) { if (back_now) set_mode(MODE_SCENES); return; }
    if (back_now) g_menu_placed = false;  // X on the menu brings it back in front
    if (!g_head_valid) return;
    if (!g_menu_placed) {
        float f[3]; forward_of(g_head_q, f);
        float l = sqrtf(f[0] * f[0] + f[2] * f[2]); if (l < 1e-3f) { f[0] = 0; f[2] = -1; l = 1; }
        g_menu_yaw = atan2f(-f[0], -f[2]);
        g_menu_at[0] = g_head_p[0] + 1.2f * f[0] / l; g_menu_at[1] = g_head_p[1] - 0.1f; g_menu_at[2] = g_head_p[2] + 1.2f * f[2] / l;
        g_menu_placed = true;
    }
    g_menu_look = -1; g_menu_hand = -1;
    static Panel panels[MAX_PANELS]; int n = menu_panels(panels);
    for (int h = 1; h >= 0; h--) {  // right hand first: it wins if both point at a panel
        g_ray_len[h] = RAY_MAX;
        if (!g_aim_valid[h]) continue;
        float d[3]; forward_of(g_aim_q[h], d);
        int hit = menu_hit(g_aim_p[h], d, &g_ray_len[h], panels, n);
        if (hit >= 0 && g_menu_look < 0) { g_menu_look = hit; g_menu_hand = h; }
    }
    bool select = a_now || (g_menu_hand >= 0 && pulled[g_menu_hand]);
    if (!select || g_menu_look < 0) return;
    const Panel* hit = &panels[g_menu_look];
    switch (hit->kind) {
    case P_MAIN: set_mode(hit->idx == 0 ? MODE_CONNECT : hit->idx == 1 ? MODE_MIRROR : MODE_SCENES); return;
    case P_TAB: g_scene_tab = hit->idx; g_scene_page = 0; return;
    case P_PREV: g_scene_page = (g_scene_page + g_scene_pages - 1) % g_scene_pages; return;
    case P_NEXT: g_scene_page = (g_scene_page + 1) % g_scene_pages; return;
    case P_MIRROR: set_mode(MODE_MIRROR); return;
    case P_CONNECT: set_mode(MODE_CONNECT); return;
    }
    pthread_mutex_lock(&g_scene_lock); int running = g_scene_current; pthread_mutex_unlock(&g_scene_lock);
    if (hit->idx == running) { set_mode(MODE_CONNECT); return; }  // the running scene: join it, no restart
    snprintf(g_loading_name, sizeof g_loading_name, "%s", hit->text);
    g_progress = g_progress_shown = 0; snprintf(g_progress_text, sizeof g_progress_text, "STOPPING");
    atomic_store(&g_loading, true); atomic_store(&g_choice, hit->idx);
    set_mode(MODE_CONNECT);
}
static void send_input(XrTime t) {
    static uint32_t seq; struct __attribute__((packed)) { char m[4]; uint32_t seq, flags; float c[8], head[7], body[JOINTS * 7], grips[14]; } p;
    XrActiveActionSet as = {g_set, XR_NULL_PATH}; XrActionsSyncInfo si = {XR_TYPE_ACTIONS_SYNC_INFO}; si.countActiveActionSets = 1; si.activeActionSets = &as;
    xrSyncActions(g_sess, &si);
    memcpy(p.m, "XINP", 4); p.seq = ++seq;
    p.c[0] = get_float(a_trig, g_hand[0]); p.c[1] = get_float(a_grip, g_hand[0]); p.c[2] = get_float(a_trig, g_hand[1]); p.c[3] = get_float(a_grip, g_hand[1]);
    for (int h = 0; h < 2; h++) {
        XrActionStateGetInfo i = {XR_TYPE_ACTION_STATE_GET_INFO}; i.action = a_stick; i.subactionPath = g_hand[h];
        XrActionStateVector2f v = {XR_TYPE_ACTION_STATE_VECTOR2F}; xrGetActionStateVector2f(g_sess, &i, &v);
        p.c[4 + h * 2] = v.isActive ? v.currentState.x : 0; p.c[5 + h * 2] = v.isActive ? v.currentState.y : 0;
    }
    XrActionStateGetInfo ai = {XR_TYPE_ACTION_STATE_GET_INFO}; ai.action = a_abtn; XrActionStateBoolean ab = {XR_TYPE_ACTION_STATE_BOOLEAN};
    xrGetActionStateBoolean(g_sess, &ai, &ab);
    g_flags = (g_flags & ~1u) | (ab.isActive && ab.currentState ? 1u : 0u);
    ai.action = a_bbtn; XrActionStateBoolean bb = {XR_TYPE_ACTION_STATE_BOOLEAN}; xrGetActionStateBoolean(g_sess, &ai, &bb);
    g_flags = (g_flags & ~16u) | (bb.isActive && bb.currentState ? 16u : 0u);
    ai.action = a_ybtn; XrActionStateBoolean yb = {XR_TYPE_ACTION_STATE_BOOLEAN}; xrGetActionStateBoolean(g_sess, &ai, &yb);
    g_flags = (g_flags & ~128u) | (yb.isActive && yb.currentState ? 128u : 0u);
    XrSpaceLocation head = {XR_TYPE_SPACE_LOCATION}; xrLocateSpace(g_view_space, g_space, t, &head);
    memcpy(p.head, &head.pose.position, 12); memcpy(p.head + 3, &head.pose.orientation, 16);
    g_head_valid = (head.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0;
    if (g_head_valid) { memcpy(g_head_p, &head.pose.position, 12); memcpy(g_head_q, &head.pose.orientation, 16); }
    ai.action = a_xbtn; XrActionStateBoolean xb = {XR_TYPE_ACTION_STATE_BOOLEAN}; xrGetActionStateBoolean(g_sess, &ai, &xb);
    for (int h = 0; h < 2; h++) {
        XrSpaceLocation al = {XR_TYPE_SPACE_LOCATION}; XrResult ar = xrLocateSpace(g_aim_space[h], g_space, t, &al);
        g_aim_valid[h] = XR_SUCCEEDED(ar) && (al.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) && (al.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);
        if (g_aim_valid[h]) { memcpy(g_aim_p[h], &al.pose.position, 12); memcpy(g_aim_q[h], &al.pose.orientation, 16); }
    }
    bool triggers[2] = {p.c[0] > 0.7f, p.c[2] > 0.7f};
    menu_input(triggers, (g_flags & 1u) != 0, xb.isActive && xb.currentState);
    int mode = atomic_load(&g_mode);
    read_body(t); p.flags = g_flags | (g_prox_far && now_s() - g_prox_far_since > 1.0 ? 8u : 0u)
        | (mode == MODE_MIRROR ? 32u : 0u) | (mode == MODE_MENU || mode == MODE_SCENES ? 64u : 0u); memcpy(p.body, g_body_out, sizeof g_body_out);
    memset(p.grips, 0, sizeof p.grips);
    for (int h = 0; h < 2; h++) {
        XrSpaceLocation g = {XR_TYPE_SPACE_LOCATION}; XrResult r = xrLocateSpace(g_grip_space[h], g_space, t, &g);
        if (XR_SUCCEEDED(r) && (g.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) && (g.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
            memcpy(p.grips + h * 7, &g.pose.position, 12); memcpy(p.grips + h * 7 + 3, &g.pose.orientation, 16);
        }
    }
    _Static_assert(sizeof p == INPUT_BYTES, "input frame size");
    if (!atomic_load(&g_pc_known)) return;
    if (g_input_udp) { sendto(g_input_fd, &p, sizeof p, 0, (struct sockaddr*)&g_pc, sizeof g_pc); return; }
    pthread_mutex_lock(&g_in_lock); memcpy(g_in_box, &p, sizeof p); g_in_full = true;
    pthread_cond_signal(&g_in_cond); pthread_mutex_unlock(&g_in_lock);
}
static void eye_init(Eye* e, const XrViewConfigurationView* v, int64_t fmt) {
    e->w = v->recommendedImageRectWidth; e->h = v->recommendedImageRectHeight;
    XrSwapchainCreateInfo c = {XR_TYPE_SWAPCHAIN_CREATE_INFO}; c.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    c.format = fmt; c.sampleCount = 1; c.width = e->w; c.height = e->h; c.faceCount = 1; c.arraySize = 1; c.mipCount = 1;
    CHECK(xrCreateSwapchain(g_sess, &c, &e->sc));
    xrEnumerateSwapchainImages(e->sc, 0, &e->n, NULL);
    e->img = calloc(e->n, sizeof *e->img); e->fbo = calloc(e->n, sizeof(GLuint));
    for (uint32_t i = 0; i < e->n; i++) e->img[i].type = XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR;
    xrEnumerateSwapchainImages(e->sc, e->n, &e->n, (XrSwapchainImageBaseHeader*)e->img);
    glGenRenderbuffers(1, &e->depth); glBindRenderbuffer(GL_RENDERBUFFER, e->depth);
    // 4x MSAA that resolves on tile: no extra colour target and little bandwidth on this GPU.
    if (pfnRbMs) pfnRbMs(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT24, e->w, e->h); else glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, e->w, e->h);
    glGenFramebuffers(e->n, e->fbo);
    for (uint32_t i = 0; i < e->n; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, e->fbo[i]);
        if (pfnFbMs) pfnFbMs(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, e->img[i].image, 0, 4);
        else glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, e->img[i].image, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, e->depth);
        GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER); if (st != GL_FRAMEBUFFER_COMPLETE) LOG("fbo incomplete %x", st);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
static void draw_box(Mat vp, Mat v_only, Mat at, float x, float y, float z, float sx, float sy, float sz,
                     float r, float g, float b) {
    Mat m = {{sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, x, y, z, 1}};
    m = mul(at, m); Mat mvp = mul(vp, m), mv = mul(v_only, m);
    glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp.m); glUniformMatrix4fv(u_mv, 1, GL_FALSE, mv.m); glUniform4f(u_color, r, g, b, 1);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
}
static const uint8_t* glyph(char c) {  // 5x7, upper case; anything else is a space
    static const uint8_t FONT[128][7] = {
        ['A'] = {14, 17, 17, 31, 17, 17, 17},
        ['B'] = {30, 17, 17, 30, 17, 17, 30},
        ['C'] = {14, 17, 16, 16, 16, 17, 14},
        ['D'] = {30, 17, 17, 17, 17, 17, 30},
        ['E'] = {31, 16, 16, 30, 16, 16, 31},
        ['F'] = {31, 16, 16, 30, 16, 16, 16},
        ['G'] = {14, 17, 16, 23, 17, 17, 15},
        ['H'] = {17, 17, 17, 31, 17, 17, 17},
        ['I'] = {14, 4, 4, 4, 4, 4, 14},
        ['J'] = {7, 2, 2, 2, 2, 18, 12},
        ['K'] = {17, 18, 20, 24, 20, 18, 17},
        ['L'] = {16, 16, 16, 16, 16, 16, 31},
        ['M'] = {17, 27, 21, 21, 17, 17, 17},
        ['N'] = {17, 25, 21, 19, 17, 17, 17},
        ['O'] = {14, 17, 17, 17, 17, 17, 14},
        ['P'] = {30, 17, 17, 30, 16, 16, 16},
        ['Q'] = {14, 17, 17, 17, 21, 18, 13},
        ['R'] = {30, 17, 17, 30, 20, 18, 17},
        ['S'] = {15, 16, 16, 14, 1, 1, 30},
        ['T'] = {31, 4, 4, 4, 4, 4, 4},
        ['U'] = {17, 17, 17, 17, 17, 17, 14},
        ['V'] = {17, 17, 17, 17, 17, 10, 4},
        ['W'] = {17, 17, 17, 21, 21, 21, 10},
        ['X'] = {17, 17, 10, 4, 10, 17, 17},
        ['Y'] = {17, 17, 10, 4, 4, 4, 4},
        ['Z'] = {31, 1, 2, 4, 8, 16, 31},
        ['0'] = {14, 17, 19, 21, 25, 17, 14},
        ['1'] = {4, 12, 4, 4, 4, 4, 14},
        ['2'] = {14, 17, 1, 2, 4, 8, 31},
        ['3'] = {31, 2, 4, 2, 1, 17, 14},
        ['4'] = {2, 6, 10, 18, 31, 2, 2},
        ['5'] = {31, 16, 30, 1, 1, 17, 14},
        ['6'] = {6, 8, 16, 30, 17, 17, 14},
        ['7'] = {31, 1, 2, 4, 8, 8, 8},
        ['8'] = {14, 17, 17, 14, 17, 17, 14},
        ['9'] = {14, 17, 17, 15, 1, 2, 12},
        [':'] = {0, 12, 12, 0, 12, 12, 0},
        ['+'] = {0, 4, 4, 31, 4, 4, 0},
        ['-'] = {0, 0, 0, 31, 0, 0, 0},
        ['='] = {0, 0, 31, 0, 31, 0, 0},
        ['('] = {2, 4, 8, 8, 8, 4, 2},
        [')'] = {8, 4, 2, 2, 2, 4, 8},
        ['_'] = {0, 0, 0, 0, 0, 0, 31},
        ['.'] = {0, 0, 0, 0, 0, 12, 12},
        ['/'] = {1, 1, 2, 4, 8, 16, 16}
    };
    if (c >= 'a' && c <= 'z') c = (char)(c - 32);
    if (c <= 0) return NULL;
    for (int r = 0; r < 7; r++) if (FONT[(int)c][r]) return FONT[(int)c];
    return NULL;
}
static void draw_text(Mat vp, Mat v_only, Mat at, const char* text, float cx, float cy, float z, float px, float r, float g, float b) {
    float x = cx - (float)strlen(text) * 6 * px / 2 + px / 2;
    for (; *text; text++, x += 6 * px) {
        const uint8_t* rows = glyph(*text); if (!rows) continue;
        for (int row = 0; row < 7; row++) for (int col = 0; col < 5; col++) if (rows[row] >> (4 - col) & 1)
            draw_box(vp, v_only, at, x + col * px, cy + (3 - row) * px, z, px * 0.45f, px * 0.45f, 0.002f, r, g, b);
    }
}
static void draw_menu(Mat vp, Mat v_only) {
    if (!g_menu_placed) return;
    float q[4] = {0, sinf(g_menu_yaw / 2), 0, cosf(g_menu_yaw / 2)};
    Mat at = from_pose(g_menu_at, q);
    glBindVertexArray(g_cube_vao);
    static Panel panels[MAX_PANELS]; int n = menu_panels(panels);
    bool scenes = atomic_load(&g_mode) == MODE_SCENES;
    int current; bool tried; pthread_mutex_lock(&g_scene_lock); current = g_scene_current; tried = g_scene_list_tried; pthread_mutex_unlock(&g_scene_lock);
    float bottom = n ? 1e9f : -0.12f; bool any_scene = false;
    for (int i = 0; i < n; i++) {
        const Panel* p = &panels[i]; any_scene |= p->kind == P_SCENE;
        bool on = g_menu_look == i, here = (p->kind == P_SCENE && p->idx == current) || (p->kind == P_TAB && p->idx == g_scene_tab);
        float k = on ? 1.0f : 0.45f;
        if (p->kind == P_MIRROR) draw_box(vp, v_only, at, p->x, p->y, 0, p->w, p->h, 0.005f, 0.75f * k + 0.1f, 0.40f * k + 0.05f, 0.05f);
        else if (here) draw_box(vp, v_only, at, p->x, p->y, 0, p->w, p->h, 0.005f, 0.05f, 0.45f * k + 0.1f, 0.25f * k);
        else draw_box(vp, v_only, at, p->x, p->y, 0, p->w, p->h, 0.005f, 0.05f, 0.25f * k + 0.05f, 0.75f * k);
        float px = p->kind == P_MAIN || p->kind == P_MIRROR || p->kind == P_CONNECT ? 0.0085f : p->kind == P_SCENE ? 0.0048f : 0.0042f, len = (float)strlen(p->text);
        if (len * 6 * px > 1.9f * p->w) px = 1.9f * p->w / (len * 6);  // a long name shrinks to fit its panel
        draw_text(vp, v_only, at, p->text, p->x, p->y, 0.012f, px, 1, 1, 1);
        if (p->y - p->h < bottom) bottom = p->y - p->h;
    }
    if (scenes) {
        draw_text(vp, v_only, at, any_scene ? "GREEN IS RUNNING: JOIN IT, OR PICK ANOTHER TO RESTART" : tried ? "THE PC OFFERS NO SCENES" : "LOOKING FOR THE PC",
                  0, 0.46f, 0.0f, 0.0065f, 0.9f, 0.9f, 0.9f);
        if (g_scene_pages > 1) {
            char page[24]; snprintf(page, sizeof page, "PAGE %d / %d", g_scene_page + 1, g_scene_pages);
            draw_text(vp, v_only, at, page, 0, bottom + 0.035f, 0.0f, 0.0055f, 0.9f, 0.9f, 0.9f);
        }
    }
    draw_text(vp, v_only, at, "X = MENU IN FRONT", 0, bottom - 0.08f, 0.0f, 0.009f, 0.9f, 0.9f, 0.9f);
    for (int h = 0; h < 2; h++) {  // a beam from each controller, brighter on the one pointing at a panel
        if (!g_aim_valid[h]) continue;
        Mat aim = from_pose(g_aim_p[h], g_aim_q[h]); float len = g_ray_len[h], k = g_menu_hand == h ? 1.0f : 0.5f;
        draw_box(vp, v_only, aim, 0, 0, -len / 2, 0.0025f, 0.0025f, len / 2, 0.3f * k, 0.8f * k, 1.0f * k);
        if (g_menu_hand == h) draw_box(vp, v_only, aim, 0, 0, -len, 0.012f, 0.012f, 0.012f, 1, 1, 1);
    }
}
// Head-locked LOADING and the scene's name, from picking a scene until the PC sends it.
static void draw_loading(Mat vp, Mat v_only) {
    if (!atomic_load(&g_loading) || !g_head_valid) return;
    Mat head = from_pose(g_head_p, g_head_q);
    glBindVertexArray(g_cube_vao);
    float pulse = 0.6f + 0.4f * (float)(fmod(now_s(), 1.0) < 0.5);
    draw_text(vp, v_only, head, "LOADING", 0, 0.10f, -1.0f, 0.009f, pulse, pulse, 1);
    draw_text(vp, v_only, head, g_loading_name, 0, 0.02f, -1.0f, 0.006f, 0.9f, 0.9f, 0.9f);
    char stage[SCENE_NAME]; float target;
    pthread_mutex_lock(&g_scene_lock); target = g_progress; memcpy(stage, g_progress_text, sizeof stage); pthread_mutex_unlock(&g_scene_lock);
    if (target > g_progress_shown) g_progress_shown += (target - g_progress_shown) * 0.05f;  // eased, never back
    const float W = 0.30f, y = -0.07f; float f = g_progress_shown < 0 ? 0 : g_progress_shown > 1 ? 1 : g_progress_shown;
    draw_box(vp, v_only, head, 0, y, -1.0f, W + 0.006f, 0.016f, 0.001f, 0.15f, 0.15f, 0.2f);
    if (f > 0.001f) draw_box(vp, v_only, head, -W + W * f, y, -0.998f, W * f, 0.011f, 0.001f, 0.3f, 0.75f, 1.0f);
    draw_text(vp, v_only, head, stage, 0, -0.13f, -1.0f, 0.005f, 0.8f, 0.8f, 0.8f);
}
// A red dot and REC up and to the left of the view while the PC records a take. Head-locked, so it is always seen.
static void draw_recording(Mat vp, Mat v_only) {
    if (!(g_status & 1u) || now_s() - g_cur_t > 1.0 || !g_head_valid) return;
    Mat head = from_pose(g_head_p, g_head_q);
    glBindVertexArray(g_cube_vao);
    float pulse = 0.6f + 0.4f * (float)(fmod(now_s(), 1.0) < 0.5);
    draw_box(vp, v_only, head, -0.30f, 0.20f, -1.0f, 0.018f, 0.018f, 0.002f, pulse, 0.05f, 0.05f);
    draw_text(vp, v_only, head, "REC", -0.215f, 0.20f, -1.0f, 0.009f, 1, 0.2f, 0.2f);
}
static void draw_eye(const XrView* view, Eye* e, float alpha) {
    uint32_t idx; XrSwapchainImageAcquireInfo ai = {XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO}; xrAcquireSwapchainImage(e->sc, &ai, &idx);
    XrSwapchainImageWaitInfo wi = {XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO}; wi.timeout = XR_INFINITE_DURATION; xrWaitSwapchainImage(e->sc, &wi);
    glBindFramebuffer(GL_FRAMEBUFFER, e->fbo[idx]); glViewport(0, 0, e->w, e->h);
    glClearColor(g_has_passthrough ? 0 : 0.05f, g_has_passthrough ? 0 : 0.06f, g_has_passthrough ? 0 : 0.08f, g_has_passthrough ? 0 : 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Mat vp = mul(projection(view->fov, 0.05f, 40.f), rigid_inverse(from_pose((float*)&view->pose.position, (float*)&view->pose.orientation)));
    Mat v_only = rigid_inverse(from_pose((float*)&view->pose.position, (float*)&view->pose.orientation));
    glUseProgram(g_prog); glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE); glDisable(GL_CULL_FACE);
    bool live = g_have_pose && now_s() - g_cur_t < 1.0;
    if (atomic_load(&g_mode) == MODE_MENU || atomic_load(&g_mode) == MODE_SCENES) {
        draw_menu(vp, v_only);
    } else if (live) {
        Mat world = sim_to_xr(g_anchor, g_yaw, g_scale);
        Mat wvp = mul(vp, world), wv = world;  // shade in room space, so the light stays put as the head moves
        for (int i = 0; i < g_num_draws; i++) {
            const Draw* d = &g_draws[i]; Mat mvp = wvp, mv = wv;
            if (d->pose >= 0) { if ((uint32_t)d->pose >= g_num_poses) continue; Mat l = link_matrix(d->pose, alpha); mvp = mul(wvp, l); mv = mul(wv, l); }
            glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp.m); glUniformMatrix4fv(u_mv, 1, GL_FALSE, mv.m);
            glUniform4f(u_color, d->rgb[0], d->rgb[1], d->rgb[2], 1);
            glBindVertexArray(d->vao); glDrawElements(GL_TRIANGLES, d->count, GL_UNSIGNED_INT, 0);
        }
    } else {  // status cube 1 m ahead on the floor: red no scene, yellow scene but no poses
        Mat m = {{0.08f, 0, 0, 0, 0, 0.08f, 0, 0, 0, 0, 0.08f, 0, 0, 0.8f, -1, 1}};
        Mat mvp = mul(vp, m), mv = mul(v_only, m);
        glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp.m); glUniformMatrix4fv(u_mv, 1, GL_FALSE, mv.m);
        glUniform4f(u_color, 1, g_num_draws ? 0.8f : 0.1f, 0.1f, 1);
        glBindVertexArray(g_cube_vao); glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
    }
    draw_recording(vp, v_only);
    if (!live) draw_loading(vp, v_only);
    glBindVertexArray(0);
    static const GLenum discard[] = {GL_DEPTH_ATTACHMENT}; glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, discard);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    XrSwapchainImageReleaseInfo ri = {XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO}; xrReleaseSwapchainImage(e->sc, &ri);
}
static void frame(void) {
    XrFrameState fs = {XR_TYPE_FRAME_STATE}; XrFrameWaitInfo w = {XR_TYPE_FRAME_WAIT_INFO}; CHECK(xrWaitFrame(g_sess, &w, &fs));
    XrFrameBeginInfo b = {XR_TYPE_FRAME_BEGIN_INFO}; CHECK(xrBeginFrame(g_sess, &b));
    poll_poses();
    uint8_t* blob = atomic_exchange(&g_blob_ready, NULL);
    if (blob) { upload_scene(blob); free(blob); }
    send_input(fs.predictedDisplayTime); play_haptics();
    XrCompositionLayerProjectionView pv[2]; XrCompositionLayerProjection proj = {XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    XrCompositionLayerPassthroughFB pl = {XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB};
    const XrCompositionLayerBaseHeader* layers[2]; uint32_t nl = 0;
    if (fs.shouldRender) {
        XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}}; XrViewState vs = {XR_TYPE_VIEW_STATE}; uint32_t nv;
        XrViewLocateInfo li = {XR_TYPE_VIEW_LOCATE_INFO}; li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO; li.displayTime = fs.predictedDisplayTime; li.space = g_space;
        CHECK(xrLocateViews(g_sess, &li, &vs, 2, &nv, views));
        float alpha = g_period > 1e-3 ? (float)((now_s() - g_cur_t) / g_period) : 1; alpha = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;
        for (int i = 0; i < 2; i++) {
            draw_eye(&views[i], &g_eye[i], alpha);
            pv[i] = (XrCompositionLayerProjectionView){XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
            pv[i].pose = views[i].pose; pv[i].fov = views[i].fov; pv[i].subImage.swapchain = g_eye[i].sc;
            pv[i].subImage.imageRect.extent.width = g_eye[i].w; pv[i].subImage.imageRect.extent.height = g_eye[i].h;
        }
        proj.space = g_space; proj.viewCount = 2; proj.views = pv;
        if (g_has_passthrough) { pl.layerHandle = g_pt_layer; pl.flags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT; layers[nl++] = (void*)&pl;
            proj.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT; }
        layers[nl++] = (void*)&proj;
        g_frames++;
    }
    XrFrameEndInfo e = {XR_TYPE_FRAME_END_INFO}; e.displayTime = fs.predictedDisplayTime; e.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE; e.layerCount = nl; e.layers = layers;
    CHECK(xrEndFrame(g_sess, &e));
    static double next_log; double t = now_s();
    if (t >= next_log) { LOG("STATUS frames=%ld pose_packets=%ld draws=%d poses=%u scale=%.2f body_frames=%ld body_valid=%d/24 state=%d", g_frames, g_pose_packets, g_num_draws, g_num_poses, g_scale, g_body_frames, g_body_valid, g_state); next_log = t + 3; }
}
// The screen is captured while the PC records a take (MainActivity.java): tell Java whenever that changes.
static void capture_follow_recording(struct android_app* app) {
    static int last = -1; static JNIEnv* env; static jmethodID set; static jclass cls;
    int on = (g_status & 1u) && now_s() - g_cur_t < 1.0;
    if (on == last) return;
    last = on;
    if (!env) {
        if ((*app->activity->vm)->AttachCurrentThread(app->activity->vm, &env, NULL) != JNI_OK) { env = NULL; return; }
        jclass local = (*env)->GetObjectClass(env, app->activity->clazz);
        cls = (*env)->NewGlobalRef(env, local);
        set = (*env)->GetStaticMethodID(env, cls, "setRecording", "(Z)V");
        if (!set) { (*env)->ExceptionClear(env); LOG("no MainActivity.setRecording: takes are not captured"); }
    }
    if (set) (*env)->CallStaticVoidMethod(env, cls, set, (jboolean)on);
    LOG("take capture %s", on ? "on" : "off");
}

void android_main(struct android_app* app) {
    char v[PROP_VALUE_MAX] = ""; if (__system_property_get("debug.telescene.pc", v) > 0) strncpy(g_pc_ip, v, sizeof g_pc_ip - 1);
    if (__system_property_get("debug.telescene.input", v) > 0 && !strcmp(v, "udp")) g_input_udp = true;
    atomic_store(&g_mode, MODE_SCENES);  // the app opens on the scene menu
    if (__system_property_get("debug.telescene.mode", v) > 0) atomic_store(&g_mode, !strcmp(v, "connect") ? MODE_CONNECT : !strcmp(v, "mirror") ? MODE_MIRROR : MODE_SCENES);
    LOG("start, PC %s, input over %s", g_pc_ip, g_input_udp ? "UDP" : "TCP");
    PFN_xrInitializeLoaderKHR init = NULL; xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR", (PFN_xrVoidFunction*)&init);
    XrLoaderInitInfoAndroidKHR li = {XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR}; li.applicationVM = app->activity->vm; li.applicationContext = app->activity->clazz; init((XrLoaderInitInfoBaseHeaderKHR*)&li);
    uint32_t nx = 0; xrEnumerateInstanceExtensionProperties(NULL, 0, &nx, NULL); XrExtensionProperties* ex = calloc(nx, sizeof *ex);
    for (uint32_t i = 0; i < nx; i++) ex[i].type = XR_TYPE_EXTENSION_PROPERTIES; xrEnumerateInstanceExtensionProperties(NULL, nx, &nx, ex);
    const char* want[] = {"XR_KHR_android_create_instance", "XR_KHR_opengl_es_enable", "XR_BD_body_tracking", "XR_EXT_local_floor", "XR_BD_controller_interaction", "XR_FB_passthrough", "XR_FB_composition_layer_alpha_blend", "XR_FB_display_refresh_rate"};
    const char* on[8]; uint32_t non = 0;
    for (int w = 0; w < 8; w++) for (uint32_t i = 0; i < nx; i++) if (!strcmp(ex[i].extensionName, want[w])) { on[non++] = want[w]; if (w == 5) g_has_passthrough = true; }
    XrInstanceCreateInfoAndroidKHR ai = {XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR}; ai.applicationVM = app->activity->vm; ai.applicationActivity = app->activity->clazz;
    XrInstanceCreateInfo ci = {XR_TYPE_INSTANCE_CREATE_INFO}; ci.next = &ai; strcpy(ci.applicationInfo.applicationName, "telescene"); ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 34);
    ci.enabledExtensionCount = non; ci.enabledExtensionNames = on;
    XrResult r = xrCreateInstance(&ci, &g_inst); LOG("xrCreateInstance = %d, %u extensions", r, non); if (XR_FAILED(r)) _exit(1);
    XrSystemGetInfo si = {XR_TYPE_SYSTEM_GET_INFO}; si.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY; XrSystemId sys; CHECK(xrGetSystem(g_inst, &si, &sys));
    PFN_xrGetOpenGLESGraphicsRequirementsKHR gr; xrGetInstanceProcAddr(g_inst, "xrGetOpenGLESGraphicsRequirementsKHR", (PFN_xrVoidFunction*)&gr);
    XrGraphicsRequirementsOpenGLESKHR req = {XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR}; gr(g_inst, sys, &req);
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY); eglInitialize(dpy, NULL, NULL);
    EGLint ca[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
    EGLConfig cfg; EGLint nc; eglChooseConfig(dpy, ca, &cfg, 1, &nc);
    EGLint xa[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE}; EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, xa);
    EGLint pa[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE}; EGLSurface pb = eglCreatePbufferSurface(dpy, cfg, pa); eglMakeCurrent(dpy, pb, pb, ctx);
    pfnFbMs = (void*)eglGetProcAddress("glFramebufferTexture2DMultisampleEXT"); pfnRbMs = (void*)eglGetProcAddress("glRenderbufferStorageMultisampleEXT");
    if (!strstr((const char*)glGetString(GL_EXTENSIONS), "GL_EXT_multisampled_render_to_texture")) pfnFbMs = NULL, pfnRbMs = NULL;
    if (!pfnFbMs || !pfnRbMs) pfnFbMs = NULL, pfnRbMs = NULL;
    LOG("GL %s, MSAA on tile %d", glGetString(GL_RENDERER), pfnFbMs != NULL);
    gl_init();
    XrGraphicsBindingOpenGLESAndroidKHR gb = {XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR}; gb.display = dpy; gb.config = cfg; gb.context = ctx;
    XrSessionCreateInfo sc = {XR_TYPE_SESSION_CREATE_INFO}; sc.next = &gb; sc.systemId = sys; r = xrCreateSession(g_inst, &sc, &g_sess); LOG("xrCreateSession = %d", r); if (XR_FAILED(r)) _exit(1);
    XrReferenceSpaceCreateInfo rs = {XR_TYPE_REFERENCE_SPACE_CREATE_INFO}; rs.poseInReferenceSpace.orientation.w = 1; rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR;
    if (XR_FAILED(xrCreateReferenceSpace(g_sess, &rs, &g_space))) { rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE; CHECK(xrCreateReferenceSpace(g_sess, &rs, &g_space)); }
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW; CHECK(xrCreateReferenceSpace(g_sess, &rs, &g_view_space));
    actions_init(); XrSessionActionSetsAttachInfo at = {XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO}; at.countActionSets = 1; at.actionSets = &g_set; CHECK(xrAttachSessionActionSets(g_sess, &at));
    for (int h = 0; h < 2; h++) {
        XrActionSpaceCreateInfo gc = {XR_TYPE_ACTION_SPACE_CREATE_INFO}; gc.action = a_grip_pose; gc.subactionPath = g_hand[h];
        gc.poseInActionSpace.orientation.w = 1; CHECK(xrCreateActionSpace(g_sess, &gc, &g_grip_space[h]));
        gc.action = a_aim_pose; CHECK(xrCreateActionSpace(g_sess, &gc, &g_aim_space[h]));
    }
    XrViewConfigurationView vv[2] = {{XR_TYPE_VIEW_CONFIGURATION_VIEW}, {XR_TYPE_VIEW_CONFIGURATION_VIEW}}; uint32_t nv;
    CHECK(xrEnumerateViewConfigurationViews(g_inst, sys, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 2, &nv, vv));
    uint32_t nf; xrEnumerateSwapchainFormats(g_sess, 0, &nf, NULL); int64_t* fm = calloc(nf, 8); xrEnumerateSwapchainFormats(g_sess, nf, &nf, fm);
    int64_t fmt = fm[0]; for (uint32_t i = 0; i < nf; i++) if (fm[i] == GL_SRGB8_ALPHA8) fmt = GL_SRGB8_ALPHA8;
    for (int i = 0; i < 2; i++) eye_init(&g_eye[i], &vv[i], fmt);
    LOG("eyes %dx%d, format %x, passthrough ext %d", g_eye[0].w, g_eye[0].h, (int)fmt, g_has_passthrough);
    passthrough_init(); body_init(); net_udp_init(); prox_init();
    pthread_t th; pthread_create(&th, NULL, start_network, NULL);
    while (!app->destroyRequested) {
        int events; struct android_poll_source* src;
        int ident; while ((ident = ALooper_pollOnce(g_running ? 0 : 100, NULL, &events, (void**)&src)) >= 0) { if (ident == PROX_LOOPER_ID) { prox_drain(); continue; } if (src) src->process(app, src); if (app->destroyRequested) break; }
        XrEventDataBuffer ev = {XR_TYPE_EVENT_DATA_BUFFER};
        while (xrPollEvent(g_inst, &ev) == XR_SUCCESS) {
            if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                g_state = ((XrEventDataSessionStateChanged*)&ev)->state; LOG("session state %d", g_state);
                if (g_state == XR_SESSION_STATE_READY) { XrSessionBeginInfo bi = {XR_TYPE_SESSION_BEGIN_INFO}; bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO; CHECK(xrBeginSession(g_sess, &bi)); g_running = true; }
                else if (g_state == XR_SESSION_STATE_STOPPING) { CHECK(xrEndSession(g_sess)); g_running = false; }
                else if (g_state == XR_SESSION_STATE_EXITING || g_state == XR_SESSION_STATE_LOSS_PENDING) { atomic_store(&g_quit, true); _exit(0); }
            }
            ev = (XrEventDataBuffer){XR_TYPE_EVENT_DATA_BUFFER};
        }
        if (g_running) frame();
        capture_follow_recording(app);
    }
    // _exit, not exit: the PICO runtime's static destructors abort on the way out of exit().
    _exit(0);
}
