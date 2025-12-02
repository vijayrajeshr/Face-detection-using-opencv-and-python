# 🧠 Real-Time Face Detection using OpenCV

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<p align="center">
  <strong>A lightweight, efficient, and beginner-friendly computer vision project that detects faces in real-time video streams.</strong>
</p>

[Report Bug](https://github.com/vijayrajeshr/Face-detection-using-opencv-and-python/issues) · [Request Feature](https://github.com/vijayrajeshr/Face-detection-using-opencv-and-python/issues)

</div>

---

## 📑 Table of Contents
- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 📖 About The Project

This project utilizes the power of **OpenCV** and pre-trained **Haar Cascade Classifiers** to identify human faces through a webcam feed. It processes video frames in real-time, converts them to grayscale for efficiency, and draws bounding boxes around detected faces. It serves as an excellent starting point for anyone interested in Computer Vision and AI.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **Real-Time Detection** | Processes live video feed instantly from the webcam. |
| **Visual Feedback** | Draws clear green rectangles `(0, 255, 0)` around identified faces. |
| **Lightweight** | Uses Haar Cascades, which are less computationally expensive than Deep Learning models. |
| **Robust Error Handling** | Includes checks for webcam availability and frame reading errors. |
| **User Controls** | Simple keyboard interaction to control the application. |

---

## 💻 Tech Stack

This project is built using the following technologies:

| Component | Technology | Version |
| :--- | :--- | :--- |
| **Language** | Python | 3.x |
| **Library** | OpenCV (`cv2`) | Latest |
| **Algorithm** | Haar Cascade | Frontal Face Default |

---

## ⚙️ Getting Started

Follow these steps to set up the project locally.

### Prerequisites
Ensure you have Python installed on your system. You can check this by running:
```bash
python --version
```

### Installation
1. **Clone the Repository**
   ```bash
   git clone [https://github.com/vijayrajeshr/Face-detection-using-opencv-and-python.git](https://github.com/vijayrajeshr/Face-detection-using-opencv-and-python.git)
   cd Face-detection-using-opencv-and-python
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   # Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install opencv-python
   ```

---

## ▶️ Usage

1.  **Run the Script Execute the main Python file to start the camera feed.**
   ```bash
    python main.py
   ```

2.**Interaction** The webcam window will open titled **"Face Detection"**.


3.Controls | Key | Action | | :---: | :--- | | q | Quit/Close the application window immediately. |


> Note: Ensure your webcam is not being used by another application (like Zoom or Teams) before running the script.

---

## 📂 Project Structure

``` plaintext
Face-detection-using-opencv-and-python/
│
├── main.py                  # The core logic script
├── README.md                # Project documentation
├── License                  # MIT License
```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project

2. Create your Feature Branch `git checkout -b feature/AmazingFeature`

3. Commit your Changes `git commit -m 'Add some AmazingFeature'`

4. Push to the Branch `git push origin feature/AmazingFeature`

5. Open a Pull Request

---

## 📜 License
Distributed under the MIT License. See **LICENSE** for more information.


   
