import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from scipy.signal import butter, filtfilt, spectrogram
from pathlib import Path 
import pandas as pd
from feature_extractor import extract_features_from_signal
from sklearn.preprocessing import LabelEncoder

# Veri seti yükleme ve ön işleme
df = pd.read_csv("ai_exports/real_features_extracted.csv")
X = df.drop(columns=["label"])
y = df["label"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Global değişkenler
last_files = []
last_fs = None
last_nyquist = None
current_signals = []

# --- Yardımcı Fonksiyonlar ---

def read_sac_file(file_path):
    st = read(file_path)
    tr = st[0]
    return tr.data, tr.stats.sampling_rate, tr.stats

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if not (0 < low < 1) or not (0 < high < 1) or not (low < high):
        return data
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_fft(y, fs):
    Y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(y), d=1/fs)
    return freq[:len(freq)//2], np.abs(Y[:len(freq)//2])

def compute_spectrogram(y, fs):
    return spectrogram(y, fs)

def get_time_axis(y, fs):
    return np.linspace(0, len(y)/fs, len(y))

def auto_suggest_filter(signal, fs):
    freq, Y = compute_fft(signal, fs)
    dominant = freq[np.argmax(Y)]
    low = max(0.1, dominant - 2)
    high = min(fs/2 - 0.1, dominant + 2)
    return round(low, 2), round(high, 2)

def update_metadata_panel(stats):
    text = (
        f"Network: {stats.network}\n"
        f"Station: {stats.station}\n"
        f"Start Time: {stats.starttime}\n"
        f"Sampling Rate: {stats.sampling_rate}"
    )
    metadata_label.config(text=text)

def on_hover(event):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        coord_label.config(text=f"x: {x:.2f}, y: {y:.2f}")

# --- Ana Fonksiyonlar ---

def reset_filter():
    if last_fs is None:
        return
    entry_fmin.delete(0, tk.END)
    entry_fmin.insert(0, "0.1")
    entry_fmax.delete(0, tk.END)
    entry_fmax.insert(0, str(round(last_nyquist - 0.1, 2)))
    analyze_sac_files(last_files)

def analyze_sac_files(file_paths):
    for widget in graph_frame.winfo_children():
        widget.destroy()

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), dpi=100)
    global current_signals
    current_signals = []

    try:
        fmin = float(entry_fmin.get())
        fmax = float(entry_fmax.get())
    except ValueError:
        messagebox.showerror("Hata", "fmin ve fmax sayısal olmalı.")
        return

    for path in file_paths:
        y, fs, stats = read_sac_file(path)
        t = get_time_axis(y, fs)
        y_filtered = bandpass_filter(y, fs, fmin, fmax)
        current_signals.append(y_filtered)

        label = path.name
        if len(label) > 25:
            label = label[:22] + "..."

        axs[0].plot(t, y_filtered, label=label)
        freq, Y = compute_fft(y_filtered, fs)
        axs[1].plot(freq, Y, label=label)
        f, tt, Sxx = compute_spectrogram(y_filtered, fs)
        axs[2].pcolormesh(tt, f, 10 * np.log10(Sxx), shading='gouraud')

    axs[0].set_title("Time Domain (Filtered)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    axs[1].set_title("Frequency Domain (FFT)")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].legend()

    axs[2].set_title("Spectrogram")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Frequency (Hz)")

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.mpl_connect("motion_notify_event", on_hover)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew")

    toolbar_frame = tk.Frame(graph_frame, bg="#1e272e")
    toolbar_frame.grid(row=1, column=0, sticky="ew")
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    graph_frame.rowconfigure(0, weight=1)
    graph_frame.columnconfigure(0, weight=1)

def browse_files():
    global last_files, last_fs, last_nyquist
    paths = filedialog.askopenfilenames(filetypes=[("SAC Files", "*.sac *.SAC")])
    if paths:
        paths = [Path(p) for p in paths]  # pathlib objesine çevir
        y0, fs0, stats0 = read_sac_file(paths[0])
        for p in paths:
            _, fs, _ = read_sac_file(p)
            if fs != fs0:
                messagebox.showerror("Hata", "Tüm dosyaların örnekleme frekansı aynı olmalı.")
                return
        last_files = paths
        last_fs = fs0
        last_nyquist = 0.5 * fs0
        entry_fmin.delete(0, tk.END)
        entry_fmin.insert(0, "0.1")
        entry_fmax.delete(0, tk.END)
        entry_fmax.insert(0, str(round(last_nyquist - 0.1, 2)))
        analyze_sac_files(paths)
        update_metadata_panel(stats0)

def export_features():
    global current_signals, last_fs
    if not current_signals:
        messagebox.showerror("Hata", "Export edilecek sinyal yok.")
        return

    features_list = []
    for signal in current_signals:
        feats = extract_features_from_signal(signal, last_fs)
        features_list.append(feats)

    cols = [
        "duration", "mean", "std", "max", "min", "rms",
        "kurtosis", "skewness", "zero_crossing_rate",
        "spectral_centroid", "peak_freq", "band_energy_1_5Hz", "band_energy_5_10Hz"
    ]
    df_export = pd.DataFrame(features_list, columns=cols)
    df_export.to_csv("ai_exports/real_features_extracted.csv", index=False)
    messagebox.showinfo("Bilgi", "Özellikler ai_exports/real_features_extracted.csv dosyasına kaydedildi.")

def apply_filter():
    if not last_files:
        messagebox.showerror("Hata", "Önce dosya seçmelisiniz.")
        return
    try:
        float(entry_fmin.get())
        float(entry_fmax.get())
    except ValueError:
        messagebox.showerror("Hata", "fmin ve fmax sayısal olmalı.")
        return
    analyze_sac_files(last_files)

# --- GUI Kurulumu ---

root = tk.Tk()
root.title("SAC Dosyası Analiz ve Filtreleme")
root.configure(bg="#1e272e")
root.geometry("1200x900")

frame_controls = tk.Frame(root, bg="#2f3640")
frame_controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

btn_browse = tk.Button(frame_controls, text="Dosya Seç", command=browse_files)
btn_browse.pack(side=tk.LEFT, padx=5)

tk.Label(frame_controls, text="fmin (Hz):", bg="#2f3640", fg="white").pack(side=tk.LEFT)
entry_fmin = tk.Entry(frame_controls, width=6)
entry_fmin.pack(side=tk.LEFT, padx=5)

tk.Label(frame_controls, text="fmax (Hz):", bg="#2f3640", fg="white").pack(side=tk.LEFT)
entry_fmax = tk.Entry(frame_controls, width=6)
entry_fmax.pack(side=tk.LEFT, padx=5)

btn_apply_filter = tk.Button(frame_controls, text="Filtreleri Uygula", command=apply_filter)
btn_apply_filter.pack(side=tk.LEFT, padx=5)

btn_reset_filter = tk.Button(frame_controls, text="Filtreyi Sıfırla", command=reset_filter)
btn_reset_filter.pack(side=tk.LEFT, padx=5)

btn_export = tk.Button(frame_controls, text="Özellikleri Export Et", command=export_features)
btn_export.pack(side=tk.LEFT, padx=5)

metadata_label = tk.Label(root, text="Dosya Metadatası", bg="#2f3640", fg="white", justify=tk.LEFT)
metadata_label.pack(side=tk.TOP, fill=tk.X, padx=5)

coord_label = tk.Label(root, text="x: -, y: -", bg="#1e272e", fg="white")
coord_label.pack(side=tk.BOTTOM, fill=tk.X)

graph_frame = tk.Frame(root, bg="#1e272e")
graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

root.mainloop()
