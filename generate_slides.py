"""
Generate PowerPoint presentation for the RAVDESS Vocal Emotion Recognition project.

Usage (after training all models):
    python generate_slides.py

Output: presentation/RAVDESS_Emotion_Recognition.pptx
"""
import os
import sys
import json

_site = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site-packages")
if os.path.isdir(_site) and _site not in sys.path:
    sys.path.insert(0, _site)

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x1B, 0x3A, 0x6B)   # dark blue
TEAL   = RGBColor(0x00, 0x8B, 0x8B)   # accent
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
LGRAY  = RGBColor(0xF4, 0xF4, 0xF4)
DGRAY  = RGBColor(0x44, 0x44, 0x44)
GREEN  = RGBColor(0x2E, 0x86, 0x48)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)

W = Inches(13.33)   # widescreen 16:9
H = Inches(7.5)


# ── Helpers ──────────────────────────────────────────────────────────────────

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)


def bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def box(slide, left, top, width, height, text="", font_size=18,
        bold=False, color=WHITE, bg_color=None, align=PP_ALIGN.LEFT,
        italic=False, wrap=True):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    if bg_color:
        fill = txBox.fill
        fill.solid()
        fill.fore_color.rgb = bg_color
    return txBox


def rect(slide, left, top, width, height, fill_color: RGBColor, alpha=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape


def img(slide, path, left, top, width=None, height=None):
    if os.path.exists(path):
        if width and height:
            slide.shapes.add_picture(path, left, top, width, height)
        elif width:
            slide.shapes.add_picture(path, left, top, width=width)
        elif height:
            slide.shapes.add_picture(path, left, top, height=height)
        else:
            slide.shapes.add_picture(path, left, top)
        return True
    return False


def load_metric(name, key="accuracy"):
    path = f"results/metrics_{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get(key, None)
    return None


def fmt(v):
    return f"{v*100:.1f}%" if v is not None else "—"


# ── Slide builders ────────────────────────────────────────────────────────────

def slide_title(prs):
    s = blank_slide(prs)
    bg(s, NAVY)
    rect(s, 0, Inches(4.8), W, Inches(2.7), TEAL)

    box(s, Inches(0.7), Inches(1.0), Inches(11.9), Inches(1.3),
        "Vocal Emotion Recognition", font_size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    box(s, Inches(0.7), Inches(2.2), Inches(11.9), Inches(0.7),
        "from Singing — RAVDESS Dataset", font_size=28, color=RGBColor(0xAD, 0xD8, 0xE6), align=PP_ALIGN.CENTER)

    box(s, Inches(0.7), Inches(3.2), Inches(11.9), Inches(0.5),
        "CNN  |  LSTM  |  GRU  |  MLP  |  SVM", font_size=18, italic=True,
        color=RGBColor(0xCC, 0xCC, 0xCC), align=PP_ALIGN.CENTER)

    box(s, Inches(0.7), Inches(5.1), Inches(11.9), Inches(0.5),
        "Machine Learning — Final Project", font_size=16, color=WHITE, align=PP_ALIGN.CENTER)
    box(s, Inches(0.7), Inches(5.6), Inches(11.9), Inches(0.5),
        "UniKore  |  A.Y. 2025/2026", font_size=14,
        color=RGBColor(0xCC, 0xCC, 0xCC), align=PP_ALIGN.CENTER)


def slide_agenda(prs):
    s = blank_slide(prs)
    bg(s, WHITE)
    rect(s, 0, 0, W, Inches(1.1), NAVY)
    box(s, Inches(0.5), Inches(0.15), Inches(12), Inches(0.8),
        "Agenda", font_size=32, bold=True, color=WHITE)

    items = [
        ("1", "Problem & Motivation",     "Why detect emotions from voice?"),
        ("2", "Dataset",                  "RAVDESS Song — 1,012 clips, 6 emotions"),
        ("3", "Audio Features",           "MFCC and Mel-Spectrograms"),
        ("4", "Models",                   "MLP, CNN, LSTM, GRU"),
        ("5", "Results & Comparison",     "Accuracy, F1, Confusion Matrices"),
        ("6", "Interpretability",         "SHAP analysis on CNN"),
        ("7", "Conclusions",              "Best model and takeaways"),
    ]

    for i, (num, title, sub) in enumerate(items):
        top = Inches(1.3 + i * 0.84)
        rect(s, Inches(0.5), top, Inches(0.5), Inches(0.6), TEAL)
        box(s, Inches(0.5), top + Pt(4), Inches(0.5), Inches(0.5),
            num, font_size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        box(s, Inches(1.2), top,          Inches(4.5), Inches(0.35),
            title, font_size=18, bold=True, color=NAVY)
        box(s, Inches(1.2), top + Inches(0.36), Inches(9), Inches(0.35),
            sub, font_size=14, color=DGRAY)


def slide_motivation(prs):
    s = blank_slide(prs)
    bg(s, WHITE)
    rect(s, 0, 0, W, Inches(1.1), NAVY)
    box(s, Inches(0.5), Inches(0.15), Inches(12), Inches(0.8),
        "Problem & Motivation", font_size=32, bold=True, color=WHITE)

    box(s, Inches(0.5), Inches(1.3), Inches(12), Inches(0.55),
        "Can a machine understand how a person feels from their voice?",
        font_size=22, bold=True, color=NAVY)

    apps = [
        ("Healthcare",    "Detect depression or anxiety from vocal patterns"),
        ("HCI",           "Voice assistants that adapt to the user's mood"),
        ("Entertainment", "Adaptive music or game soundtracks"),
        ("Security",      "Stress detection in call centres"),
    ]
    for i, (title, desc) in enumerate(apps):
        col = i % 2
        row = i // 2
        left = Inches(0.5 + col * 6.4)
        top  = Inches(2.1 + row * 2.3)
        rect(s, left, top, Inches(6.0), Inches(2.0), LGRAY)
        rect(s, left, top, Inches(6.0), Inches(0.45), TEAL)
        box(s, left + Inches(0.1), top + Inches(0.05), Inches(5.8), Inches(0.38),
            title, font_size=16, bold=True, color=WHITE)
        box(s, left + Inches(0.1), top + Inches(0.5), Inches(5.8), Inches(1.3),
            desc, font_size=15, color=DGRAY)


def slide_dataset(prs):
    s = blank_slide(prs)
    bg(s, WHITE)
    rect(s, 0, 0, W, Inches(1.1), NAVY)
    box(s, Inches(0.5), Inches(0.15), Inches(12), Inches(0.8),
        "Dataset — RAVDESS Song", font_size=32, bold=True, color=WHITE)

    # Left: key stats
    stats = [
        ("1,012", "audio clips"),
        ("24",    "actors (12M + 12F)"),
        ("6",     "emotion classes"),
        ("22,050", "Hz sample rate"),
        ("4 sec", "clip duration"),
    ]
    for i, (val, label) in enumerate(stats):
        top = Inches(1.3 + i * 1.1)
        rect(s, Inches(0.4), top, Inches(2.3), Inches(0.95), LGRAY)
        box(s, Inches(0.4), top + Inches(0.05), Inches(2.3), Inches(0.5),
            val, font_size=28, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
        box(s, Inches(0.4), top + Inches(0.52), Inches(2.3), Inches(0.35),
            label, font_size=13, color=DGRAY, align=PP_ALIGN.CENTER)

    # Right: emotion table
    emotions = [
        ("neutral",  "01", "Very few clips — class imbalance"),
        ("calm",     "02", "Relaxed, soft singing"),
        ("happy",    "03", "Energetic, bright tone"),
        ("sad",      "04", "Slow, low pitch"),
        ("angry",    "05", "Loud, high energy"),
        ("fearful",  "06", "Tense, irregular"),
    ]
    box(s, Inches(3.0), Inches(1.25), Inches(4), Inches(0.4),
        "Emotion Classes", font_size=16, bold=True, color=NAVY)
    for i, (name, code, desc) in enumerate(emotions):
        top = Inches(1.65 + i * 0.93)
        rect(s, Inches(3.0), top, Inches(0.55), Inches(0.75), TEAL)
        box(s, Inches(3.0), top + Inches(0.1), Inches(0.55), Inches(0.5),
            code, font_size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        box(s, Inches(3.65), top,               Inches(1.5),  Inches(0.4),
            name.capitalize(), font_size=15, bold=True, color=NAVY)
        box(s, Inches(3.65), top + Inches(0.38), Inches(3.5), Inches(0.38),
            desc, font_size=12, color=DGRAY)

    # Distribution image
    img_path = "results/eda_class_distribution.png"
    if not img(s, img_path, Inches(7.5), Inches(1.2), width=Inches(5.5)):
        box(s, Inches(7.5), Inches(2.5), Inches(5.5), Inches(3.5),
            "[Run notebook 01_EDA.ipynb to generate chart]",
            font_size=14, color=DGRAY, align=PP_ALIGN.CENTER)


def slide_features(prs):
    s = blank_slide(prs)
    bg(s, WHITE)
    rect(s, 0, 0, W, Inches(1.1), NAVY)
    box(s, Inches(0.5), Inches(0.15), Inches(12), Inches(0.8),
        "Audio Feature Extraction", font_size=32, bold=True, color=WHITE)

    # MFCC
    rect(s, Inches(0.4), Inches(1.2), Inches(6.0), Inches(5.8), LGRAY)
    rect(s, Inches(0.4), Inches(1.2), Inches(6.0), Inches(0.45), NAVY)
    box(s, Inches(0.5), Inches(1.25), Inches(5.8), Inches(0.38),
        "MFCC  (Mel-Frequency Cepstral Coefficients)", font_size=16, bold=True, color=WHITE)
    mfcc_text = (
        "40 coefficients per time frame\n"
        "Captures timbre and vocal tract shape\n"
        "Input for RNN/LSTM and MLP\n"
        "Shape per clip: (173 frames x 40 coefficients)"
    )
    box(s, Inches(0.5), Inches(1.75), Inches(5.8), Inches(1.5),
        mfcc_text, font_size=14, color=DGRAY)
    img(s, "results/eda_mfcc.png", Inches(0.5), Inches(3.2), height=Inches(3.6))

    # Mel-spectrogram
    rect(s, Inches(6.8), Inches(1.2), Inches(6.1), Inches(5.8), LGRAY)
    rect(s, Inches(6.8), Inches(1.2), Inches(6.1), Inches(0.45), TEAL)
    box(s, Inches(6.9), Inches(1.25), Inches(5.9), Inches(0.38),
        "Mel-Spectrogram", font_size=16, bold=True, color=WHITE)
    mel_text = (
        "128 mel frequency bands\n"
        "Frequency vs time 'image'\n"
        "Input for CNN (treated as 2D image)\n"
        "Shape per clip: (1 x 128 bands x 173 frames)"
    )
    box(s, Inches(6.9), Inches(1.75), Inches(5.9), Inches(1.5),
        mel_text, font_size=14, color=DGRAY)
    img(s, "results/eda_mel_spectrograms.png", Inches(6.9), Inches(3.2), height=Inches(3.6))


def slide_models_overview(prs):
    s = blank_slide(prs)
    bg(s, WHITE)
    rect(s, 0, 0, W, Inches(1.1), NAVY)
    box(s, Inches(0.5), Inches(0.15), Inches(12), Inches(0.8),
        "Models Overview", font_size=32, bold=True, color=WHITE)

    models = [
        ("MLP",  "Baseline",        DGRAY,
         "Input: mean+std of MFCC\n80-dim vector\n2 hidden layers\n(256 -> 128 -> 6)"),
        ("SVM",  "Classical ML",    RGBColor(0x5D, 0x6D, 0x7E),
         "Input: mean+std of MFCC\nRBF kernel\nClass-weighted\nScikit-learn"),
        ("CNN",  "Deep Learning",   TEAL,
         "Input: mel-spectrogram\n4 conv blocks\nGlobal Avg Pooling\n45,830 parameters"),
        ("LSTM", "Deep Learning",   NAVY,
         "Input: MFCC sequence\n2-layer Bi-LSTM\nHidden size: 128\nLast timestep output"),
    ]

    for i, (name, category, color, desc) in enumerate(models):
        left = Inches(0.35 + i * 3.2)
        rect(s, left, Inches(1.2), Inches(3.0), Inches(5.8), LGRAY)
        rect(s, left, Inches(1.2), Inches(3.0), Inches(0.9), color)
        box(s, left, Inches(1.25), Inches(3.0), Inches(0.55),
            name, font_size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        box(s, left, Inches(1.82), Inches(3.0), Inches(0.3),
            category, font_size=12, italic=True, color=RGBColor(0xDD,0xDD,0xDD),
            align=PP_ALIGN.CENTER)
        box(s, left + Inches(0.1), Inches(2.3), Inches(2.8), Inches(3.5),
            desc, font_size=14, color=DGRAY)

    box(s, Inches(0.5), Inches(6.8), Inches(12.5), Inches(0.4),
        "All models use speaker-independent evaluation: actors 21-24 reserved for test",
        font_size=13, italic=True, color=DGRAY, align=PP_ALIGN.CENTER)


def slide_results(prs):
    s = blank_slide(prs)
    bg(s, WHITE)
    rect(s, 0, 0, W, Inches(1.1), NAVY)
    box(s, Inches(0.5), Inches(0.15), Inches(12), Inches(0.8),
        "Results — Model Comparison", font_size=32, bold=True, color=WHITE)

    model_names = ["mlp", "cnn", "lstm", "gru"]
    display_names = ["MLP", "CNN", "LSTM", "GRU"]
    colors = [DGRAY, TEAL, NAVY, RGBColor(0x8E, 0x44, 0xAD)]

    # Bar chart area — show image if available
    if not img(s, "results/model_comparison.png", Inches(6.8), Inches(1.2), height=Inches(5.8)):
        box(s, Inches(6.8), Inches(3.5), Inches(6.0), Inches(1.0),
            "[Run notebooks to generate chart]", font_size=14, color=DGRAY, align=PP_ALIGN.CENTER)

    # Metrics table (left side)
    headers = ["Model", "Accuracy", "F1 Macro", "F1 Weighted"]
    col_widths = [Inches(1.1), Inches(1.3), Inches(1.3), Inches(1.5)]
    col_starts = [Inches(0.3), Inches(1.4), Inches(2.7), Inches(4.0)]

    top = Inches(1.3)
    for j, (header, left) in enumerate(zip(headers, col_starts)):
        rect(s, left, top, col_widths[j], Inches(0.45), NAVY)
        box(s, left, top + Inches(0.03), col_widths[j], Inches(0.4),
            header, font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    best_acc = max(
        (load_metric(n, "accuracy") or 0) for n in model_names
    )

    for i, (name, display, color) in enumerate(zip(model_names, display_names, colors)):
        row_top = Inches(1.75 + i * 0.6)
        row_bg = LGRAY if i % 2 == 0 else WHITE
        acc = load_metric(name, "accuracy")
        f1m = load_metric(name, "f1_macro")
        f1w = load_metric(name, "f1_weighted")
        vals = [display, fmt(acc), fmt(f1m), fmt(f1w)]

        is_best = acc is not None and abs(acc - best_acc) < 0.001
        highlight = color if is_best else row_bg
        txt_color = WHITE if is_best else DGRAY

        for j, (val, left) in enumerate(zip(vals, col_starts)):
            rect(s, left, row_top, col_widths[j], Inches(0.52),
                 color if (is_best and j == 0) else row_bg)
            box(s, left, row_top + Inches(0.05), col_widths[j], Inches(0.45),
                val, font_size=14,
                bold=(j == 0),
                color=WHITE if (is_best and j == 0) else DGRAY,
                align=PP_ALIGN.CENTER)

    if best_acc:
        box(s, Inches(0.3), Inches(4.3), Inches(6.0), Inches(0.4),
            f"Best model accuracy: {fmt(best_acc)}",
            font_size=15, bold=True, color=GREEN)


def slide_confusion(prs):
    s = blank_slide(prs)
    bg(s, WHITE)
    rect(s, 0, 0, W, Inches(1.1), NAVY)
    box(s, Inches(0.5), Inches(0.15), Inches(12), Inches(0.8),
        "Confusion Matrices", font_size=32, bold=True, color=WHITE)

    positions = [
        ("mlp",  "MLP",  Inches(0.2),  Inches(1.2)),
        ("cnn",  "CNN",  Inches(3.55), Inches(1.2)),
        ("lstm", "LSTM", Inches(6.9),  Inches(1.2)),
        ("gru",  "GRU",  Inches(10.2), Inches(1.2)),
    ]
    for name, label, left, top in positions:
        box(s, left, top, Inches(3.0), Inches(0.38),
            label, font_size=16, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        if not img(s, f"results/cm_{name}.png", left, top + Inches(0.35), width=Inches(3.1)):
            rect(s, left, top + Inches(0.35), Inches(3.1), Inches(5.5), LGRAY)
            box(s, left, top + Inches(2.8), Inches(3.1), Inches(0.5),
                "Run training first", font_size=12, color=DGRAY, align=PP_ALIGN.CENTER)


def slide_shap(prs):
    s = blank_slide(prs)
    bg(s, WHITE)
    rect(s, 0, 0, W, Inches(1.1), NAVY)
    box(s, Inches(0.5), Inches(0.15), Inches(12), Inches(0.8),
        "Interpretability — SHAP Analysis", font_size=32, bold=True, color=WHITE)

    box(s, Inches(0.5), Inches(1.2), Inches(12), Inches(0.45),
        "Which regions of the mel-spectrogram most influence the CNN's prediction?",
        font_size=18, bold=True, color=NAVY)

    box(s, Inches(0.5), Inches(1.75), Inches(5.5), Inches(2.8),
        "SHAP (SHapley Additive exPlanations)\n\n"
        "Red regions = frequency bands that strongly push toward the predicted emotion\n\n"
        "Blue regions = areas that suppress the prediction\n\n"
        "Allows us to understand what the CNN 'hears'",
        font_size=15, color=DGRAY)

    if not img(s, "results/shap_cnn.png", Inches(6.0), Inches(1.2), height=Inches(5.8)):
        rect(s, Inches(6.0), Inches(1.2), Inches(7.0), Inches(5.8), LGRAY)
        box(s, Inches(6.0), Inches(3.8), Inches(7.0), Inches(0.5),
            "[Run notebook 05_evaluation.ipynb]",
            font_size=14, color=DGRAY, align=PP_ALIGN.CENTER)


def slide_conclusions(prs):
    s = blank_slide(prs)
    bg(s, NAVY)
    rect(s, 0, 0, W, Inches(1.1), TEAL)
    box(s, Inches(0.5), Inches(0.15), Inches(12), Inches(0.8),
        "Conclusions", font_size=32, bold=True, color=WHITE)

    best_model = max(
        ["mlp", "cnn", "lstm", "gru"],
        key=lambda n: load_metric(n, "f1_macro") or 0
    )
    best_acc = fmt(load_metric(best_model, "accuracy"))
    best_f1  = fmt(load_metric(best_model, "f1_macro"))

    findings = [
        f"Best model: {best_model.upper()} — Accuracy {best_acc}, F1 macro {best_f1}",
        "CNN excels by treating audio as a 2D image (mel-spectrogram)",
        "LSTM/GRU capture temporal dynamics of vocal emotion effectively",
        "SHAP reveals which frequency bands are most discriminative per emotion",
        "Neutral class remains hardest to classify (fewest training samples)",
    ]
    for i, text in enumerate(findings):
        top = Inches(1.3 + i * 1.1)
        rect(s, Inches(0.5), top + Inches(0.15), Inches(0.08), Inches(0.5), TEAL)
        box(s, Inches(0.8), top, Inches(11.5), Inches(0.9),
            text, font_size=18, color=WHITE)

    box(s, Inches(0.5), Inches(7.0), Inches(12), Inches(0.35),
        "Future work: transformer-based models (Wav2Vec 2.0), cross-corpus evaluation",
        font_size=14, italic=True, color=RGBColor(0xAA, 0xCC, 0xCC), align=PP_ALIGN.CENTER)


def slide_thankyou(prs):
    s = blank_slide(prs)
    bg(s, NAVY)
    rect(s, 0, Inches(2.8), W, Inches(1.9), TEAL)

    box(s, 0, Inches(0.8), W, Inches(1.5),
        "Thank You", font_size=54, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    box(s, 0, Inches(2.95), W, Inches(0.7),
        "Questions?", font_size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    box(s, 0, Inches(5.0), W, Inches(0.5),
        "GitHub: github.com/anastasia-drojjina/ravdess-emotion-recognition",
        font_size=15, color=RGBColor(0xAD, 0xD8, 0xE6), align=PP_ALIGN.CENTER)
    box(s, 0, Inches(5.6), W, Inches(0.5),
        "Dataset: RAVDESS — Livingstone & Russo (2018) — doi:10.5281/zenodo.1188976",
        font_size=13, italic=True, color=RGBColor(0x88, 0xAA, 0xBB), align=PP_ALIGN.CENTER)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("presentation", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    prs = new_prs()

    print("Building slides...")
    slide_title(prs)          ; print("  1/9  Title")
    slide_agenda(prs)         ; print("  2/9  Agenda")
    slide_motivation(prs)     ; print("  3/9  Motivation")
    slide_dataset(prs)        ; print("  4/9  Dataset")
    slide_features(prs)       ; print("  5/9  Features")
    slide_models_overview(prs); print("  6/9  Models")
    slide_results(prs)        ; print("  7/9  Results")
    slide_confusion(prs)      ; print("  8/9  Confusion matrices")
    slide_shap(prs)           ; print("  9/9  SHAP")
    slide_conclusions(prs)    ; print(" 10/10 Conclusions")
    slide_thankyou(prs)       ; print(" 11/11 Thank you")

    out = "presentation/RAVDESS_Emotion_Recognition.pptx"
    prs.save(out)
    print(f"\nSaved -> {out}")
    print("Open in PowerPoint or Google Slides to fine-tune fonts and spacing.")


if __name__ == "__main__":
    main()
