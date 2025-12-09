import os
import threading
import numpy as np
import traceback
from tkinter import Tk, StringVar, PhotoImage, filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Auto-detect project root and assets
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fruit_quality_mobilenetv2.h5")
VAL_PATH = os.path.join(PROJECT_ROOT, "dataset", "val")

# load labels
def load_labels_from_val_folder(val_path):
    if not os.path.exists(val_path):
        return []
    labels = [d for d in sorted(os.listdir(val_path)) if os.path.isdir(os.path.join(val_path, d))]
    # Keep human-friendly form for display; internal use uses model outputs order assumption
    display_labels = [lbl.replace("_", " ").title() for lbl in labels]
    # If no val folder, fallback to a common ordering
    if not labels:
        labels = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]
        display_labels = [lbl.replace("_", " ").title() for lbl in labels]
    return labels, display_labels

RAW_LABELS, DISPLAY_LABELS = load_labels_from_val_folder(VAL_PATH)

# Load model (in background to not block GUI)
_model = None
_model_load_error = None

def _load_model_background():
    global _model, _model_load_error
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
    except Exception as e:
        _model_load_error = str(e)
        _model = None

# Start loading model in background thread immediately
threading.Thread(target=_load_model_background, daemon=True).start()


# GUI
class FruitApp:
    def __init__(self, root):
        self.root = root
        root.title("Fruit Quality Detector")
        root.geometry("720x520")
        root.resizable(False, False)
        self.style = ttk.Style()
        # Set theme if available
        try:
            self.style.theme_use('clam')
        except:
            pass
        # Configure custom styles
        self.style.configure("Card.TFrame", background="#F6F7FB")
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("Sub.TLabel", font=("Segoe UI", 10))
        self.style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        self.style.map("Big.TButton",
                       foreground=[("active", "#ffffff")],
                       background=[("active", "#007ACC"), ("!active", "#0a74da")])
        self.style.configure("Danger.TLabel", foreground="#C62828", font=("Segoe UI", 10, "bold"))

        # Frames
        main = ttk.Frame(root, padding=12)
        main.pack(fill="both", expand=True)

        header_frame = ttk.Frame(main)
        header_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(header_frame, text="Fruit Quality Detector", style="Header.TLabel").pack(side="left")
        self.theme_var = StringVar(value="Light")
        self.theme_btn = ttk.Button(header_frame, text="Dark Mode", command=self.toggle_theme)
        self.theme_btn.pack(side="right")

        body = ttk.Frame(main, style="Card.TFrame", padding=10)
        body.pack(fill="both", expand=True)

        # image preview
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 12))
        self.preview_canvas = ttk.Label(left, text="No image", anchor="center", background="#ffffff", width=42)
        self.preview_canvas.pack(pady=(0, 8))
        # Use a placeholder image area - will set image via PhotoImage
        self.preview_img = None

        # Buttons
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=(6, 0))
        self.choose_btn = ttk.Button(btn_frame, text="Choose Image", style="Big.TButton", command=self.choose_image)
        self.choose_btn.pack(fill="x", pady=(0, 6))
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", style="Big.TButton", command=self.analyze_image, state="disabled")
        self.analyze_btn.pack(fill="x")
        self.info_label = ttk.Label(left, text="Model status: loading...", style="Sub.TLabel", wraplength=250)
        self.info_label.pack(pady=(8,0))

        # results
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Prediction", style="Sub.TLabel", anchor="w").pack(anchor="w")
        self.result_title = ttk.Label(right, text="No prediction yet", font=("Segoe UI", 14, "bold"))
        self.result_title.pack(anchor="w", pady=(4, 8))

        # Top-3 list
        self.top_frame = ttk.Frame(right)
        self.top_frame.pack(fill="x", pady=(4, 4))
        self.top_labels = []
        for i in range(3):
            lbl = ttk.Label(self.top_frame, text=f"{i+1}. —", font=("Segoe UI", 11))
            lbl.pack(anchor="w", pady=6)
            self.top_labels.append(lbl)

        # Confidence bar
        ttk.Label(right, text="Confidence", style="Sub.TLabel").pack(anchor="w", pady=(8,0))
        self.conf_progress = ttk.Progressbar(right, orient="horizontal", length=360, mode="determinate")
        self.conf_progress.pack(anchor="w", pady=(6,10))
        self.conf_value_label = ttk.Label(right, text="")
        self.conf_value_label.pack(anchor="w")

        # Status / spinner
        self.spinner = ttk.Progressbar(right, mode="indeterminate", length=200)
        self.spinner.pack(anchor="center", pady=(12, 0))
        self.spinner.stop()

        # Error label
        self.err_label = ttk.Label(right, text="", style="Danger.TLabel")
        self.err_label.pack(anchor="w", pady=(6,0))

        # Internal state
        self.current_image_path = None
        self.last_probs = None

        # After init, start a periodic check for model loaded
        self.root.after(500, self._check_model_loaded)

    def toggle_theme(self):
        # Basic light/dark swap by changing background colors
        if self.theme_var.get() == "Light":
            self.theme_var.set("Dark")
            self.style.configure("Card.TFrame", background="#1f2937")
            self.style.configure("Header.TLabel", foreground="#ffffff")
            self.style.configure("Sub.TLabel", foreground="#e5e7eb")
            self.preview_canvas.configure(background="#374151", foreground="#e5e7eb")
            self.theme_btn.config(text="Light Mode")
        else:
            self.theme_var.set("Light")
            self.style.configure("Card.TFrame", background="#F6F7FB")
            self.style.configure("Header.TLabel", foreground="#000000")
            self.style.configure("Sub.TLabel", foreground="#6b7280")
            self.preview_canvas.configure(background="#ffffff", foreground="#000000")
            self.theme_btn.config(text="Dark Mode")

    def _check_model_loaded(self):
        global _model, _model_load_error
        if _model is None and _model_load_error is None:
            # still loading
            self.info_label.config(text="Model status: loading...")
            self.root.after(500, self._check_model_loaded)
            return
        if _model is None and _model_load_error:
            self.info_label.config(text=f"Model status: error")
            self.err_label.config(text=f"Model load error: {_model_load_error}")
            self.analyze_btn.config(state="disabled")
        else:
            self.info_label.config(text="Model status: ready")
            self.analyze_btn.config(state="normal" if self.current_image_path else "disabled")
            self.err_label.config(text="")
        # do not schedule further checks after success/failure

    def choose_image(self):
        path = filedialog.askopenfilename(title="Choose an image",
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.current_image_path = path
        self._display_preview(path)
        # enable analyze if model is ready
        if _model is not None:
            self.analyze_btn.config(state="normal")
        else:
            self.analyze_btn.config(state="disabled")

    def _display_preview(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((360, 360))
            self._tkimg = ImageTk.PhotoImage(img)
            self.preview_canvas.configure(image=self._tkimg, text="")
        except Exception as e:
            self.preview_canvas.configure(text=f"Error loading image\n{e}")

    def analyze_image(self):
        if not self.current_image_path:
            self.err_label.config(text="No image selected.")
            return
        if _model is None:
            self.err_label.config(text="Model not loaded yet.")
            return

        # start spinner and run prediction in a thread
        self.err_label.config(text="")
        self.spinner.start(10)
        self.analyze_btn.config(state="disabled")
        self.choose_btn.config(state="disabled")
        threading.Thread(target=self._run_prediction_thread, daemon=True).start()

    def _run_prediction_thread(self):
        try:
            probs = self._predict(self.current_image_path)  # numpy probs
            # sort top3
            indices = np.argsort(probs)[::-1][:3]
            top = [(i, probs[i]) for i in indices]
            # update UI in main thread
            self.root.after(0, self._update_results_ui, top)
        except Exception as e:
            tb = traceback.format_exc()
            self.root.after(0, self._show_error, str(e) + "\n" + tb)
        finally:
            self.root.after(0, self._prediction_done)

    def _prediction_done(self):
        self.spinner.stop()
        self.analyze_btn.config(state="normal")
        self.choose_btn.config(state="normal")

    def _show_error(self, text):
        self.err_label.config(text=text[:300])  # show truncated error

    def _predict(self, img_path):
        # Preprocess image same as training
        img = load_img(img_path, target_size=(224, 224))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        preds = _model.predict(arr)[0]  # probabilities
        # If model outputs single neuron (binary) convert to two-class array
        if preds.ndim == 0 or preds.size == 1:
            # transform sigmoid output to two-element prob array [1-p, p]
            p = float(preds)
            preds = np.array([1.0 - p, p])
        # Ensure length matches labels; if not, fallback to DISPLAY_LABELS length
        if preds.shape[0] != len(RAW_LABELS):
            # Attempt to map by sorted order (best-effort)
            # If counts mismatch, create generic labels
            # We won't crash; just create dummy labels
            padded = np.zeros(max(len(preds), len(RAW_LABELS)))
            padded[:len(preds)] = preds
            preds = padded
        return preds

    def _update_results_ui(self, top):
        # top: list of (index, prob)
        # If RAW_LABELS empty fallback to DISPLAY_LABELS
        labels = RAW_LABELS if RAW_LABELS else [l.lower().replace(" ", "") for l in DISPLAY_LABELS]
        for i, (idx, prob) in enumerate(top):
            name = labels[idx] if idx < len(labels) else f"Class {idx}"
            display_name = name.replace("_", " ").title()
            self.top_labels[i].config(text=f"{i+1}. {display_name} — {prob*100:.2f}%")
        # set main title as top1
        top1 = top[0]
        top1_name = (labels[top1[0]] if top1[0] < len(labels) else f"Class {top1[0]}").replace("_", " ").title()
        self.result_title.config(text=f"{top1_name}")
        conf = top1[1] * 100
        self.conf_progress['value'] = conf
        self.conf_value_label.config(text=f"{conf:.2f}%")

# run app
def main():
    root = Tk()
    app = FruitApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
