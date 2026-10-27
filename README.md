# 🌿 Plant Disease Recognition System (MobileNetV2)

An AI-powered computer vision system capable of identifying **38 different classes** of plant leaf diseases using **MobileNetV2 Transfer Learning** and an interactive web dashboard built with **Streamlit**.

---

## 🚀 Features

- **MobileNetV2 Transfer Learning:** Pre-trained on ImageNet and fine-tuned on ~87,000 augmented leaf images from the PlantVillage dataset.
- **Ultra Lightweight & Fast:** Only **~24 MB** model size (compared to ~90 MB for standard CNNs) with sub-second CPU inference.
- **38 Classes Supported:** Detects healthy leaves and diseases across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
- **⚡ 1-Click Setup & Launch:** Windows launcher scripts (`run.bat` and `setup.bat`) automatically verify 64-bit Python, upgrade pip, install dependencies, and launch the dashboard.
- **Sample Test Images Included:** Comes pre-bundled with real leaf images in `sample_test_images/` for instant testing.

---

## 📂 Project Structure

```text
├── main.py                             # Streamlit dashboard application (MobileNetV2 engine)
├── mobilenet_v2_plant_disease.keras    # Pre-trained MobileNetV2 weights (~24 MB)
├── download_model.py                   # Model weight verification & auto-downloader
├── install_dependencies.py             # Smart environment & 64-bit Python checker
├── run.bat                             # 1-Click setup & launch script for Windows
├── setup.bat                           # 1-Click environment setup script
├── requirements.txt                    # Project dependencies
├── home_page.jpeg                      # Cover banner for the dashboard
├── Train_plant_disease_mobilenetv2.ipynb # MobileNetV2 transfer learning notebook
├── Train_plant_disease.ipynb           # Original Custom CNN notebook
├── Test_plant_disease.ipynb            # Original evaluation notebook
├── sample_test_images/                 # Real test leaf images for quick testing
│   ├── Apple_Cedar_apple_rust.JPG
│   ├── Apple_healthy.JPG
│   ├── Corn_Common_rust.JPG
│   └── Tomato_Early_blight.JPG
├── .gitignore                          # Git ignore configuration
└── README.md                           # Documentation
```

---

## ⚡ 1-Click Setup & Run (Windows)

Simply **double-click** `run.bat` in File Explorer, or open your terminal and run:

```cmd
run.bat
```

What `run.bat` does automatically:
1. Verifies that you are running **64-bit Python**.
2. Upgrades `pip` to the latest release.
3. Installs all required packages (`streamlit`, `tensorflow`, `numpy`, `pillow`, etc.).
4. Ensures the pre-trained MobileNetV2 weights (`mobilenet_v2_plant_disease.keras`) are ready.
5. Launches the Streamlit dashboard in your default browser at `http://localhost:8501`.

---

## 🧠 Model Architecture & Comparison

| Specification | Custom CNN (Previous) | MobileNetV2 (Current) |
| :--- | :--- | :--- |
| **Model File** | `trained_plant_disease_model.keras` | `mobilenet_v2_plant_disease.keras` |
| **Model Size** | **~90 MB** | **~24 MB** (approx. 4× smaller) |
| **Parameters** | ~7.8 Million | ~2.3 Million |
| **Pre-training** | Trained from scratch | Pre-trained on ImageNet |
| **Input Resolution** | $128 \times 128$ | $224 \times 224$ |
| **Architecture** | 5-stage standard Conv2D + Dense | Inverted Residuals & Depthwise Convolutions |

---

## 🛠️ Manual Setup

If you prefer using a Python virtual environment manually:

```bash
# 1. Clone repository
git clone https://github.com/Karthicknaruto/new-one.git
cd new-one

# 2. Create and activate virtual environment (64-bit Python 3.10-3.12 recommended)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch web application
streamlit run main.py
```

---

## 📄 License
This project is for educational and research purposes.
