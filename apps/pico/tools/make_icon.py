"""Draw the launcher icon, res/mipmap-xxxhdpi/ic_launcher.png: a robot head wearing a headset's visor, with a
signal arc, on a deep blue tile. Run from apps/pico; needs Pillow."""
from PIL import Image, ImageDraw

S, K = 512, 4  # output size, supersampling
N = S * K
img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
bg = Image.new("RGBA", (N, N))
top, bottom = (24, 44, 92), (10, 120, 140)
px = bg.load()
for y in range(N):
    for x in range(0, N, 1):
        t = (x + y) / (2 * N)
        px[x, y] = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)) + (255,)
mask = Image.new("L", (N, N), 0)
ImageDraw.Draw(mask).rounded_rectangle((0, 0, N - 1, N - 1), radius=int(N * 0.22), fill=255)
img.paste(bg, (0, 0), mask)
d = ImageDraw.Draw(img)
u = N / 512
def box(x0, y0, x1, y1, r, fill):
    d.rounded_rectangle((x0 * u, y0 * u, x1 * u, y1 * u), radius=r * u, fill=fill)
# neck and shoulders
box(222, 352, 290, 400, 10, (225, 232, 240, 255))
box(130, 388, 382, 470, 46, (240, 244, 248, 255))
# head
box(136, 132, 376, 362, 64, (248, 250, 252, 255))
# visor (the headset) and its strap
box(116, 196, 396, 286, 40, (18, 26, 44, 255))
box(150, 214, 362, 268, 26, (30, 210, 230, 255))
d.rounded_rectangle((176 * u, 226 * u, 336 * u, 256 * u), radius=15 * u, fill=(150, 245, 255, 255))
# signal arcs, top right
for i, r in enumerate((44, 78, 112)):
    w = int(16 * u)
    c = (392 * u, 120 * u)
    d.arc((c[0] - r * u, c[1] - r * u, c[0] + r * u, c[1] + r * u), start=200, end=250 + 0 * i, fill=(150, 245, 255, 255 - 50 * i), width=w)
out = img.resize((S, S), Image.LANCZOS)
out.save("res/mipmap-xxxhdpi/ic_launcher.png")
print("res/mipmap-xxxhdpi/ic_launcher.png")
