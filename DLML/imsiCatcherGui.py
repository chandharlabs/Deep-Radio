import tkinter as tk
import subprocess

# ---------------- Functions ----------------

def run_kal_scan():
    subprocess.Popen([
        "x-terminal-emulator", "-e",
        "bash", "-i", "-c", "scanGSM"
    ])

def run_grgsm():
    subprocess.Popen([
        "x-terminal-emulator", "-e",
        "grgsm_livemon -g 40 -f 937.2e6"
    ])


def kill_imsi_catcher():
    # Find PIDs of any running simple_IMSI-catcher.py scripts
    try:
        pids = subprocess.check_output(
            ["pgrep", "-f", "simple_IMSI-catcher.py"]
        ).decode().split()
        for pid in pids:
            subprocess.run(["kill", "-9", pid])
    except subprocess.CalledProcessError:
        # No process found, ignore
        pass


def run_imsi():
    kill_imsi_catcher()
    # Run the IMSI catcher in its own working directory to avoid FileNotFoundError
    imsi_path = "/home/wiguy/IMSI-catcher"
    subprocess.Popen([
        "x-terminal-emulator", "-e",
        f"bash -c 'cd {imsi_path} && python3 simple_IMSI-catcher.py; exec bash'"
    ])


def run_wireshark():
    # GSM analysis display filter
    display_filter = "gsmtap || lapdm"

    subprocess.Popen([
        "x-terminal-emulator", "-e",
        "bash", "-i", "-c",
        f"sudo wireshark -i any -k -Y '{display_filter}'; exec bash"
    ])


# ---------------- GUI ----------------

root = tk.Tk()
root.title("Deep Radio")
root.geometry("460x360")
root.configure(bg="#0f172a")  # dark blue background

# Company Name
company_label = tk.Label(
    root,
    text="Deep Radio: GSM IMSI Catcher",
    font=("Helvetica", 16, "bold"),
    fg="#38bdf8",
    bg="#0f172a"
)
company_label.pack(pady=(15, 5))

# App Title
title = tk.Label(
    root,
    text="Chandhar Research Labs Pvt. Ltd.",
    font=("Courier", 14),
    fg="white",
    bg="#0f172a"
)
title.pack(pady=(0, 20))

# Button frame
btn_frame = tk.Frame(root, bg="#0f172a")
btn_frame.pack()

btn_scan = tk.Button(
    btn_frame,
    text="📡 Scan GSM Channels",
    width=30,
    bg="#2563eb",
    fg="white",
    font=("Arial", 11, "bold"),
    command=run_kal_scan
)
btn_scan.pack(pady=6)

btn_grgsm = tk.Button(
    btn_frame,
    text="📶 Select GSM Channel",
    width=30,
    bg="#16a34a",
    fg="white",
    font=("Arial", 11, "bold"),
    command=run_grgsm
)
btn_grgsm.pack(pady=6)

btn_imsi = tk.Button(
    btn_frame,
    text="🛰️ Catch IMSI",
    width=30,
    bg="#dc2626",
    fg="white",
    font=("Arial", 11, "bold"),
    command=run_imsi
)
btn_imsi.pack(pady=6)


btn_wireshark = tk.Button(
    btn_frame,
    text="🦈 Start Wireshark",
    width=30,
    bg="#7c3aed",
    fg="white",
    font=("Arial", 11, "bold"),
    command=run_wireshark
)
btn_wireshark.pack(pady=6)



# Exit button
exit_btn = tk.Button(
    root,
    text="Exit",
    width=20,
    bg="#334155",
    fg="white",
    command=root.quit
)
exit_btn.pack(pady=20)




root.mainloop()

