# 🧾 Receipt Intelligence Engine

> **A Detection-First OCR Pipeline for Complex Receipt Understanding**

This project is an end-to-end solution for extracting structured data (JSON) from unstructured receipt images. Unlike traditional OCR approaches, we utilize a **Detection-First Strategy** using **YOLOv8** to localize key regions (Merchant, Total, Date, Line Items) before applying text recognition, ensuring high accuracy on complex datasets like **CORD** and **SROIE**.

---

## 📂 Project Structure

The repository is organized to separate **experimental code** (Notebooks) from **production code** (Src).

```text
receipt-intelligence-engine/
│
├── 📂 .venv/                  # Managed by uv (Do not edit manually)
│
├── 📂 data/                   # 🛑 IGNORED BY GIT. Store datasets here locally.
│   ├── cord/                  # CORD Dataset (Raw images & JSONs)
│   ├── sroie_v2/              # SROIE Dataset (Raw images & TXT)
│   └── local/                 # Local/Private datasets
│
├── 📂 notebooks/              # 🧪 Experimentation Lab
│   ├── 01_preprocessing.ipynb # Member 1: Image filters & deskewing tests
│   ├── 02_detection_prep.ipynb# Member 2: Data conversion (JSON -> YOLO)
│   └── ...
│
├── 📂 src/                    # 🏭 Production Code (Reusable Modules)
│   ├── 📂 preprocessing/      # (Member 1)
│   │   ├── filters.py         # Grayscale, Denoising, Thresholding functions
│   │   └── geometry.py        # Skew correction & Perspective transforms
│   │
│   ├── 📂 detection/          # (Member 2)
│   │   ├── model.py           # YOLOv8 Inference logic
│   │   └── dataset.py         # Data loaders for CORD/SROIE
│   │
│   ├── 📂 ocr/                # (Member 2 & 3)
│   │   └── engine.py          # Wrapper for EasyOCR/PaddleOCR
│   │
│   ├── 📂 parsing/            # (Member 3)
│   │   └── extractor.py       # Regex & Post-processing logic
│   │
│   └── 📂 api/                # (Member 4)
│       └── main.py            # FastAPI Backend Entry Point
│
├── .gitignore                 # Prevents data & junk files from being uploaded
├── pyproject.toml             # Project dependencies list
├── uv.lock                    # Exact version locking (ensures consistency)
└── README.md                  # Project Documentation

```

---

## ⚡ Setup & Dependency Management (`uv`)

We use **[uv](https://github.com/astral-sh/uv)** for extremely fast package management. This ensures every team member has the **exact same environment**.

### 1️⃣ Installation (First Time Only)

If you don't have `uv` installed, run this in your terminal:

**Windows (PowerShell):**

```powershell
irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex

```

**Mac/Linux:**

```bash
curl -lsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

```

### 2️⃣ Sync Environment (For Team Members)

After cloning the repository, you don't need to manually install `opencv` or `torch`. Just run:

```bash
uv sync

```

*This command reads `uv.lock` and creates a `.venv` with all required libraries automatically.*

### 3️⃣ Running Code

To run any Python script using the project's environment, prefix the command with `uv run`:

```bash
# Example: Running the API
uv run python src/api/main.py

# Example: Running a script
uv run python src/preprocessing/test.py

```

### 4️⃣ Adding New Libraries

If you need to add a new library (e.g., `matplotlib`), do **not** use pip. Use:

```bash
uv add matplotlib

```

*This updates `pyproject.toml` and `uv.lock` so other team members get it next time they run `uv sync`.*

---

## 🚫 Data Privacy & Git Rules

1. **NEVER push data to GitHub.** The `.gitignore` is configured to ignore the `data/` folder.
2. **Dataset Sharing:** Datasets (SROIE, CORD, Local) are shared via **Google Drive**. Download them and place them in the `data/` folder following the structure above.
3. **Notebooks vs. Src:**
* Use **Notebooks** for visualization and trial & error.
* Move working logic to **`src/`** functions immediately.
* The API (`main.py`) can only import from `src/`, not from notebooks.



---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Vision:** OpenCV, Scikit-Image
* **Detection:** Ultralytics YOLOv8
* **OCR:** EasyOCR / PaddleOCR
* **Backend:** FastAPI
* **Data Processing:** Pandas, NumPy
---
