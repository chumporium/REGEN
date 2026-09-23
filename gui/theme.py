"""REGEN look and feel: uses the built-in Windows style (windows11 / windowsvista) so it feels like a regular
desktop application, plus Windows glyph icons and a neutral matplotlib plot style."""
import matplotlib
from cycler import cycler
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QStyleFactory

ACCENT = "#1f5f99"          # neutral blue (titles, main plot lines)
INK = "#1f1f1f"
MUTED = "#5f6368"
BORDER = "#d0d0d0"
WARN = "#b45309"
BAD = "#c42b1c"
GOOD = "#107c10"

# plot series colors: classic palette (similar to matplotlib/Excel)
PLOT_COLORS = ["#1f5f99", "#e8710a", "#2e7d32", "#c62828", "#6a1b9a", "#5d4037", "#00838f", "#757575"]

# Segoe MDL2 Assets / Segoe Fluent Icons glyph codes (available on Windows 10 & 11)
GLYPHS = {"home": "", "data": "", "eval": "", "analysis": "", "ann": "",
          "graph": "", "optim": "", "nsga": "", "predict": "", "unc": "",
          "norm": "", "cases": "", "open": "", "save": "", "new": "",
          "import": "", "excel": "", "pdf": "", "menu": "", "info": "",
          "help": "", "saveas": "", "search": "", "check": "", "warn": "",
          "notes": "", "coef": "", "scatter": "", "summary": "",
          "play": "", "stop": "", "add": "", "remove": "", "copy": ""}


def icon_font(size=12):
    f = QFont()
    f.setFamilies(["Segoe Fluent Icons", "Segoe MDL2 Assets"])
    f.setPointSizeF(size)
    return f


def glyph_icon(name, color=INK, size=20):
    """QIcon from a Windows icon font glyph (monochrome, like the built-in Windows icons)."""
    if color == "white":                        # buttons no longer have a colored background -> dark icon
        color = INK
    pm = QPixmap(QSize(size * 2, size * 2))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    f = icon_font()
    f.setPixelSize(int(size * 1.2))
    p.setFont(f)
    p.setPen(QColor(color))
    p.drawText(QRect(0, 0, size * 2, size * 2), Qt.AlignCenter, GLYPHS.get(name, name))
    p.end()
    pm.setDevicePixelRatio(2.0)
    return QIcon(pm)


# only a few tweaks; everything else follows the built-in Windows style
STYLE = """
QTreeWidget#Nav { border: none; background: palette(window); }
QTreeWidget#Nav::item { padding: 3px 2px; }
QLabel#StartTitle { font-size: 16pt; }
"""


def apply(app):
    keys = [k.lower() for k in QStyleFactory.keys()]
    for name in ("windows11", "windowsvista", "fusion"):
        if name in keys:
            app.setStyle(name)
            break
    try:                                        # reports & plots are designed for a light background
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    except AttributeError:
        pass
    app.setFont(QFont("Segoe UI", 9))
    app.setStyleSheet(STYLE)


def apply_matplotlib():
    matplotlib.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "axes.prop_cycle": cycler(color=PLOT_COLORS),
        "axes.titleweight": "normal",
        "axes.titlesize": 10,
        "figure.facecolor": "white",
        "image.cmap": "viridis",
        "legend.fontsize": 8,
    })
