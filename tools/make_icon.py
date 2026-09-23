"""Creates assets/icon.ico (a simple stylized response surface contour)."""
import os

from PIL import Image, ImageDraw

SIZE = 256
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "icon.ico")


def draw():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([8, 8, SIZE - 8, SIZE - 8], radius=48, fill=(31, 95, 168, 255))
    colors = [(64, 140, 210), (96, 180, 200), (140, 210, 170), (200, 230, 140), (250, 225, 110)]
    cx, cy = 150, 110
    for i, col in enumerate(colors):
        rx, ry = 105 - i * 20, 80 - i * 15
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col + (255,),
                  outline=(20, 50, 90, 255), width=3)
    for x, y in [(60, 196), (128, 196), (196, 196), (60, 128), (60, 60)]:
        d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=(220, 40, 40, 255), outline="white", width=3)
    return img


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    draw().save(OUT, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("Icon created:", os.path.abspath(OUT))
