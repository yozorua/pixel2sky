"""Interactive Alt/Az sky-coordinate viewer — pixel2sky demo app.

Lets you load any sky image, tune camera parameters, overlay Alt/Az grid
lines, and query coordinates in both directions (pixel ↔ sky).

Requirements
------------
    pip install PyQt6 matplotlib

Usage
-----
    python examples/sky_viewer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")  # must be called before pyplot

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pixel2sky import SkyMapper
from pixel2sky.projection import EquidistantFisheye, Rectilinear, StereographicFisheye

# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib canvas widget
# ──────────────────────────────────────────────────────────────────────────────

class _Canvas(FigureCanvasQTAgg):
    """Thin wrapper that exposes a single `ax` attribute."""

    def __init__(self) -> None:
        self.fig = Figure(facecolor="#111827")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, facecolor="#111827")
        self.ax.text(
            0.5, 0.5, "No image loaded",
            ha="center", va="center",
            transform=self.ax.transAxes,
            color="#6b7280", fontsize=14,
        )
        self.ax.axis("off")
        self.fig.tight_layout(pad=0.1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


# ──────────────────────────────────────────────────────────────────────────────
# Small UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def _spin(
    lo: float, hi: float, val: float,
    decimals: int = 1, step: float = 1.0,
) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setFixedWidth(90)
    return s


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: #374151;")
    return f


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

class SkyViewerApp(QMainWindow):
    """Main application window."""

    # Grid overlay colours
    _ALT_COLOUR  = "#34d399"  # teal-green  — iso-altitude lines
    _AZ_COLOUR   = "#60a5fa"  # sky-blue    — iso-azimuth lines

    # Query result colours — muted, easy on the eyes
    _COL_PIX2SKY = "#7dd3fc"  # light cyan  — pixel → sky
    _COL_SKY2PIX = "#a5b4fc"  # soft indigo — sky → pixel

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SkyViewer")
        self.resize(1300, 800)

        self._image: np.ndarray | None = None
        self._mapper: SkyMapper | None = None
        self._grid_artists: list = []
        self._query_markers: list = []

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # ── Left control panel ────────────────────────────────────────────
        left_widget = QWidget()
        left_widget.setMaximumWidth(290)
        left_widget.setMinimumWidth(260)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(8, 8, 8, 8)
        splitter.addWidget(left_widget)

        # Open image button
        load_btn = QPushButton("  Open Image…")
        load_btn.setMinimumHeight(38)
        load_btn.setStyleSheet(
            "QPushButton { background:#1d4ed8; color:white;"
            "  border-radius:4px; font-weight:bold; }"
            "QPushButton:hover { background:#2563eb; }"
        )
        load_btn.clicked.connect(self._load_image)
        left_layout.addWidget(load_btn)

        left_layout.addWidget(_hline())

        # ── Camera parameters ─────────────────────────────────────────────
        cam_box = QGroupBox("Camera Parameters")
        cam_form = QFormLayout(cam_box)
        cam_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        cam_form.setVerticalSpacing(4)

        # Projection model — dropdown
        self._proj_combo = QComboBox()
        self._proj_combo.addItem("Rectilinear")
        self._proj_combo.addItem("Equidistant Fisheye")
        self._proj_combo.addItem("Stereographic Fisheye  ✦")
        self._proj_combo.setCurrentIndex(1)
        self._proj_combo.setToolTip(
            "Rectilinear  — standard pinhole, r = f·tan(θ)\n"
            "Equidistant  — fisheye, r = f·θ  (most common)\n"
            "Stereographic ✦ — fisheye, r = 2f·tan(θ/2), angle-preserving"
        )
        cam_form.addRow("Model:", self._proj_combo)

        # Focal length — switchable mode
        self._focal_mode = QComboBox()
        self._focal_mode.addItems(["Plate scale  (″/px)", "Physical  (mm + μm)"])
        self._focal_mode.currentIndexChanged.connect(self._on_focal_mode_changed)
        cam_form.addRow("Focal input:", self._focal_mode)

        # Mode 0: plate scale
        self._lbl_plate      = QLabel("Plate scale (″/px):")
        self._sp_plate_scale = _spin(0.01, 99999.0, 258.0, 2, 10.0)
        self._sp_plate_scale.setToolTip(
            "Angular size of one pixel in arcseconds.\n"
            "Converts to f = 206 265 / scale  (px)."
        )
        cam_form.addRow(self._lbl_plate, self._sp_plate_scale)

        # Mode 1: physical parameters
        self._lbl_focal_mm = QLabel("Focal length (mm):")
        self._sp_focal_mm  = _spin(0.1, 10000.0, 8.5, 2, 0.5)
        self._lbl_pixel_um = QLabel("Pixel size (μm):")
        self._sp_pixel_um  = _spin(0.1, 100.0, 3.76, 2, 0.1)
        self._sp_focal_mm.setToolTip("Lens focal length in millimetres.")
        self._sp_pixel_um.setToolTip(
            "Physical size of one sensor pixel in micrometres (μm).\n"
            "Converts to f = focal_mm × 1000 / pixel_μm  (px)."
        )
        cam_form.addRow(self._lbl_focal_mm, self._sp_focal_mm)
        cam_form.addRow(self._lbl_pixel_um, self._sp_pixel_um)

        # Hide physical rows until mode 1 is selected
        self._lbl_focal_mm.hide()
        self._sp_focal_mm.hide()
        self._lbl_pixel_um.hide()
        self._sp_pixel_um.hide()

        self._sp_az0  = _spin(-360.0, 360.0,  0.0, 1, 5.0)
        self._sp_alt0 = _spin( -90.0,  90.0, 45.0, 1, 5.0)
        self._sp_roll = _spin(-180.0, 180.0,  0.0, 1, 5.0)
        self._sp_cx   = _spin(0.0, 1e6, 960.0, 1, 1.0)
        self._sp_cy   = _spin(0.0, 1e6, 540.0, 1, 1.0)

        cam_form.addRow("Az₀ (°):",  self._sp_az0)
        cam_form.addRow("Alt₀ (°):", self._sp_alt0)
        cam_form.addRow("Roll (°):", self._sp_roll)
        cam_form.addRow("Xc (px):",  self._sp_cx)
        cam_form.addRow("Yc (px):",  self._sp_cy)

        apply_btn = QPushButton("Apply Parameters")
        apply_btn.setStyleSheet(
            "QPushButton { background:#065f46; color:white; border-radius:4px; }"
            "QPushButton:hover { background:#047857; }"
        )
        apply_btn.clicked.connect(self._apply_params)
        cam_form.addRow(apply_btn)

        left_layout.addWidget(cam_box)

        # ── Grid overlay ──────────────────────────────────────────────────
        grid_box = QGroupBox("Alt / Az Grid Overlay")
        grid_form = QFormLayout(grid_box)
        grid_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        grid_form.setVerticalSpacing(4)

        self._sp_az_step  = _spin(1.0, 90.0, 30.0, 0, 5.0)
        self._sp_alt_step = _spin(1.0, 90.0, 15.0, 0, 5.0)
        grid_form.addRow("Az step (°):",  self._sp_az_step)
        grid_form.addRow("Alt step (°):", self._sp_alt_step)

        self._grid_toggle = QCheckBox("Show Grid")
        self._grid_toggle.setChecked(False)
        self._grid_toggle.stateChanged.connect(self._on_grid_toggle)
        grid_form.addRow(self._grid_toggle)

        left_layout.addWidget(grid_box)

        # ── Coordinate query ──────────────────────────────────────────────
        q_box = QGroupBox("Coordinate Query")
        q_layout = QVBoxLayout(q_box)
        q_layout.setSpacing(4)

        # Pixel → Sky
        hdr1 = QLabel("Pixel  →  Sky")
        hdr1.setFont(QFont("", weight=QFont.Weight.Bold))
        hdr1.setStyleSheet(f"color: {self._COL_PIX2SKY};")
        q_layout.addWidget(hdr1)

        p_row = QWidget()
        prl = QHBoxLayout(p_row)
        prl.setContentsMargins(0, 0, 0, 0)
        prl.addWidget(QLabel("x:"))
        self._sp_qx = _spin(0.0, 1e6, 0.0, 1)
        prl.addWidget(self._sp_qx)
        prl.addWidget(QLabel("y:"))
        self._sp_qy = _spin(0.0, 1e6, 0.0, 1)
        prl.addWidget(self._sp_qy)
        q_layout.addWidget(p_row)

        p2s_go = QPushButton("→  Get Alt / Az")
        p2s_go.clicked.connect(self._query_pixel_to_sky)
        q_layout.addWidget(p2s_go)

        self._lbl_p2s = QLabel("alt: —     az: —")
        self._lbl_p2s.setStyleSheet(
            f"color:{self._COL_PIX2SKY}; font-weight:bold; padding:2px;"
        )
        self._lbl_p2s.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        q_layout.addWidget(self._lbl_p2s)

        q_layout.addWidget(_hline())

        # Sky → Pixel
        hdr2 = QLabel("Sky  →  Pixel")
        hdr2.setFont(QFont("", weight=QFont.Weight.Bold))
        hdr2.setStyleSheet(f"color: {self._COL_SKY2PIX};")
        q_layout.addWidget(hdr2)

        s_row = QWidget()
        srl = QHBoxLayout(s_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.addWidget(QLabel("alt:"))
        self._sp_qalt = _spin(-90.0, 90.0, 45.0, 1, 1.0)
        srl.addWidget(self._sp_qalt)
        srl.addWidget(QLabel("az:"))
        self._sp_qaz = _spin(0.0, 360.0, 0.0, 1, 1.0)
        srl.addWidget(self._sp_qaz)
        q_layout.addWidget(s_row)

        s2p_go = QPushButton("→  Get Pixel x / y")
        s2p_go.clicked.connect(self._query_sky_to_pixel)
        q_layout.addWidget(s2p_go)

        self._lbl_s2p = QLabel("x: —     y: —")
        self._lbl_s2p.setStyleSheet(
            f"color:{self._COL_SKY2PIX}; font-weight:bold; padding:2px;"
        )
        self._lbl_s2p.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        q_layout.addWidget(self._lbl_s2p)

        left_layout.addWidget(q_box)

        # Click hint
        hint = QLabel("Tip: left-click the image to query pixel → sky")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280; font-size:10px; padding:2px;")
        left_layout.addWidget(hint)
        left_layout.addStretch()

        # ── Status bar ────────────────────────────────────────────────────
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Open an image to begin.")

        # ── Right canvas panel ────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(2, 2, 2, 2)
        rl.setSpacing(0)

        self._canvas = _Canvas()
        self._toolbar = NavigationToolbar2QT(self._canvas, right)
        rl.addWidget(self._toolbar)
        rl.addWidget(self._canvas)
        splitter.addWidget(right)
        splitter.setSizes([270, 1030])

        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image file", str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.tiff *.bmp);;All files (*)",
        )
        if not path:
            return
        try:
            img = mpimg.imread(path)
        except Exception as exc:
            self._status.showMessage(f"Error loading image: {exc}")
            return

        if img.dtype.kind == "u":
            img = img.astype(np.float32) / float(np.iinfo(img.dtype).max)
        self._image = img
        h, w = img.shape[:2]

        self._sp_cx.setValue(w / 2.0)
        self._sp_cy.setValue(h / 2.0)

        self._redraw_image()
        self._apply_params()
        self._status.showMessage(f"Loaded: {Path(path).name}  ({w} × {h} px)")

    # ── Focal length helpers ──────────────────────────────────────────────────

    def _on_focal_mode_changed(self, idx: int) -> None:
        plate = (idx == 0)
        self._lbl_plate.setVisible(plate)
        self._sp_plate_scale.setVisible(plate)
        self._lbl_focal_mm.setVisible(not plate)
        self._sp_focal_mm.setVisible(not plate)
        self._lbl_pixel_um.setVisible(not plate)
        self._sp_pixel_um.setVisible(not plate)

    def _focal_to_pixels(self) -> float:
        if self._focal_mode.currentIndex() == 0:
            scale = max(self._sp_plate_scale.value(), 1e-9)
            return 206265.0 / scale
        f_mm   = self._sp_focal_mm.value()
        pix_um = max(self._sp_pixel_um.value(), 1e-9)
        return f_mm * 1000.0 / pix_um

    # ── Parameter application ─────────────────────────────────────────────────

    def _apply_params(self) -> None:
        if self._image is None:
            self._status.showMessage("Load an image first.")
            return
        h, w = self._image.shape[:2]
        try:
            f    = self._focal_to_pixels()
            az0  = self._sp_az0.value()
            alt0 = self._sp_alt0.value()
            roll = self._sp_roll.value()
            cx   = self._sp_cx.value()
            cy   = self._sp_cy.value()

            idx = self._proj_combo.currentIndex()
            if idx == 0:
                proj       = Rectilinear(focal_length=f)
                model_name = "Rectilinear"
            elif idx == 2:
                proj       = StereographicFisheye(focal_length=f)
                model_name = "Stereographic Fisheye"
            else:
                proj       = EquidistantFisheye(focal_length=f)
                model_name = "Equidistant Fisheye"

            self._mapper = SkyMapper(
                image_width=w, image_height=h,
                projection=proj,
                az0=az0, alt0=alt0, roll=roll,
                cx=cx, cy=cy,
            )
            fov_h, fov_v = self._mapper.fov_degrees()
            self._status.showMessage(
                f"{model_name}  |  f={f:.1f} px  |  "
                f"az={az0:.1f}°  alt={alt0:.1f}°  roll={roll:.1f}°  |  "
                f"FOV: {fov_h:.1f}° × {fov_v:.1f}°"
            )
        except Exception as exc:
            self._status.showMessage(f"Parameter error: {exc}")
            self._mapper = None
            return

        # Refresh grid automatically if it is currently shown
        if self._grid_toggle.isChecked():
            self._show_grid()

    # ── Image drawing ─────────────────────────────────────────────────────────

    def _redraw_image(self) -> None:
        ax = self._canvas.ax
        ax.clear()
        if self._image is not None:
            ax.imshow(self._image, origin="upper")
        ax.axis("off")
        self._canvas.fig.tight_layout(pad=0.1)
        self._canvas.draw_idle()
        self._grid_artists.clear()
        self._query_markers.clear()

    # ── Grid overlay ──────────────────────────────────────────────────────────

    def _on_grid_toggle(self, state: int) -> None:
        if state == Qt.CheckState.Checked.value:
            self._show_grid()
        else:
            self._clear_grid()

    def _show_grid(self) -> None:
        if self._mapper is None:
            self._status.showMessage("Apply parameters first.")
            self._grid_toggle.blockSignals(True)
            self._grid_toggle.setChecked(False)
            self._grid_toggle.blockSignals(False)
            return
        self._clear_grid()

        az_step  = self._sp_az_step.value()
        alt_step = self._sp_alt_step.value()
        n = 720

        ax = self._canvas.ax

        # Iso-altitude lines: constant alt, sweep az 0 → 360°
        for alt_val in np.arange(-90.0, 90.0 + alt_step * 0.5, alt_step):
            az_s = np.linspace(0.0, 360.0, n + 1)
            x, y = self._mapper.altaz_to_pixel(np.full(n + 1, alt_val), az_s)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 2:
                continue
            xp = np.where(valid, x, np.nan)
            yp = np.where(valid, y, np.nan)
            (ln,) = ax.plot(xp, yp, color=self._ALT_COLOUR, lw=0.9, alpha=0.80)
            self._grid_artists.append(ln)
            idx = np.where(valid)[0]
            mi = idx[len(idx) // 2]
            t = ax.text(
                xp[mi], yp[mi], f"{alt_val:.0f}°",
                color=self._ALT_COLOUR, fontsize=7, ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#00000099", lw=0),
            )
            self._grid_artists.append(t)

        # Iso-azimuth lines: constant az, sweep alt −90 → 90°
        for az_val in np.arange(0.0, 360.0, az_step):
            alt_s = np.linspace(-90.0, 90.0, n)
            x, y = self._mapper.altaz_to_pixel(alt_s, np.full(n, az_val))
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 2:
                continue
            xp = np.where(valid, x, np.nan)
            yp = np.where(valid, y, np.nan)
            (ln,) = ax.plot(xp, yp, color=self._AZ_COLOUR, lw=0.9, alpha=0.80)
            self._grid_artists.append(ln)
            idx = np.where(valid)[0]
            mi = idx[len(idx) // 2]
            t = ax.text(
                xp[mi], yp[mi], f"{az_val:.0f}°",
                color=self._AZ_COLOUR, fontsize=7, ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#00000099", lw=0),
            )
            self._grid_artists.append(t)

        self._canvas.draw_idle()

    def _clear_grid(self) -> None:
        for artist in self._grid_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._grid_artists.clear()
        self._canvas.draw_idle()

    # ── Coordinate queries ────────────────────────────────────────────────────

    def _clear_markers(self) -> None:
        for a in self._query_markers:
            try:
                a.remove()
            except ValueError:
                pass
        self._query_markers.clear()

    def _place_marker(self, x: float, y: float, colour: str) -> None:
        self._clear_markers()
        ax = self._canvas.ax
        (dot,) = ax.plot(x, y, "+", ms=16, mew=2.0, color=colour, zorder=10)
        ring = mpatches.Circle(
            (x, y), radius=10, fill=False, edgecolor=colour, lw=1.5, zorder=10,
        )
        ax.add_patch(ring)
        self._query_markers.extend([dot, ring])
        self._canvas.draw_idle()

    def _query_pixel_to_sky(self) -> None:
        if self._mapper is None:
            self._lbl_p2s.setText("No mapper — apply parameters first.")
            return
        x, y = self._sp_qx.value(), self._sp_qy.value()
        alt, az = self._mapper.pixel_to_altaz(x, y)
        alt_f, az_f = float(alt), float(az)
        if np.isfinite(alt_f) and np.isfinite(az_f):
            self._lbl_p2s.setText(f"alt: {alt_f:.4f}°     az: {az_f:.4f}°")
            self._place_marker(x, y, self._COL_PIX2SKY)
        else:
            self._lbl_p2s.setText("Outside valid projection range.")

    def _query_sky_to_pixel(self) -> None:
        if self._mapper is None:
            self._lbl_s2p.setText("No mapper — apply parameters first.")
            return
        alt, az = self._sp_qalt.value(), self._sp_qaz.value()
        x, y = self._mapper.altaz_to_pixel(alt, az)
        xf, yf = float(x), float(y)
        if np.isfinite(xf) and np.isfinite(yf):
            self._lbl_s2p.setText(f"x: {xf:.2f}     y: {yf:.2f}")
            self._place_marker(xf, yf, self._COL_SKY2PIX)
        else:
            self._lbl_s2p.setText("Point outside sensor bounds.")

    def _on_canvas_click(self, event) -> None:  # type: ignore[override]
        if event.inaxes is not self._canvas.ax:
            return
        if event.button != 1 or self._mapper is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._sp_qx.setValue(event.xdata)
        self._sp_qy.setValue(event.ydata)
        self._query_pixel_to_sky()


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor("#1f2937"))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor("#f9fafb"))
    palette.setColor(QPalette.ColorRole.Base,            QColor("#111827"))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#1f2937"))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor("#374151"))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor("#f9fafb"))
    palette.setColor(QPalette.ColorRole.Text,            QColor("#f9fafb"))
    palette.setColor(QPalette.ColorRole.Button,          QColor("#374151"))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#f9fafb"))
    palette.setColor(QPalette.ColorRole.BrightText,      QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor("#2563eb"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    win = SkyViewerApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
