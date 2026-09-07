"""
EyeSpeak Launcher
Simple desktop dashboard to launch the lip-reading and eye-tracking modules
without typing commands each time.

Place this file in the same folder as your other scripts (gashcodes/),
then just double-click it (or run: python launcher.py) to open the dashboard.
"""

import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox

# ============================== CONFIG ====================================
# Update these filenames if you rename your scripts later.
LIP_READING_SCRIPT = "realtime_inference_v2.py"
EYE_TRACKING_SCRIPT = "eye_cursor_pro_fixed_no_emergency_button_v2_clean.py"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EyeSpeak Launcher")
        self.root.geometry("480x380")
        self.root.minsize(420, 320)
        self.root.resizable(True, True)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        header = ttk.Frame(root, padding=20)
        header.pack(fill="x")
        ttk.Label(header, text="EyeSpeak", font=("Segoe UI", 22, "bold")).pack()
        ttk.Label(header, text="Choose a module to launch",
                  font=("Segoe UI", 10)).pack(pady=(4, 0))

        body = ttk.Frame(root, padding=(20, 10, 20, 20))
        body.pack(fill="both", expand=True)

        button_style = ttk.Style()
        button_style.configure("Big.TButton", font=("Segoe UI", 13))

        ttk.Button(
            body, text="🎤  Lip Reading", style="Big.TButton",
            command=lambda: self.launch(LIP_READING_SCRIPT)
        ).pack(fill="x", pady=10, ipady=16)

        ttk.Button(
            body, text="👁  Eye Tracking", style="Big.TButton",
            command=lambda: self.launch(EYE_TRACKING_SCRIPT)
        ).pack(fill="x", pady=10, ipady=16)

        self.status_var = tk.StringVar(value="Ready to control system handsfree ?")
        ttk.Label(body, textvariable=self.status_var,
                  font=("Segoe UI", 9)).pack(pady=(15, 0))

    def launch(self, script_name):
        script_path = os.path.join(SCRIPT_DIR, script_name)

        if not os.path.isfile(script_path):
            messagebox.showerror(
                "File not found",
                f"Couldn't find:\n{script_path}\n\n"
                f"Check the filename in launcher.py matches your actual script."
            )
            return

        try:
            # sys.executable ensures we use the SAME Python interpreter
            # (and venv) that launcher.py itself is running under.
            subprocess.Popen([sys.executable, script_path], cwd=SCRIPT_DIR)
            self.status_var.set(f"Launched: {script_name}")
        except Exception as e:
            messagebox.showerror("Launch failed", str(e))
            self.status_var.set("Launch failed.")


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()