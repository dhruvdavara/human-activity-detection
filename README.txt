HUMAN ACTIVITY DETECTION - LOCAL INFERENCE

This package uses the trained MobileNetV3-Small + LSTM checkpoint
from the Colab training notebook.

Classes:
0 Boxing
1 Handclapping
2 Handwaving
3 Jogging
4 Running

Input:
- Video file readable by OpenCV (AVI/MP4/etc.)
- 16 uniformly sampled frames
- 224x224 RGB
- ImageNet normalization

SETUP IN VS CODE / POWERSHELL

1. Open this folder in VS Code.

2. Create a virtual environment:
   python -m venv .venv

3. Activate it:
   .venv\Scripts\Activate.ps1

4. If PowerShell blocks activation, use:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .venv\Scripts\Activate.ps1

5. Install dependencies:
   pip install -r requirements.txt

6. Put a test video in the folder, or use any path to a video.

7. Run:
   python test.py "path\to\your\video.avi"

Example:
   python test.py "test_video.avi"

IMPORTANT:
The checkpoint has 6 output neurons because that is how the model
was trained. The sixth output is not a project class, so inference
uses only outputs 0-4, corresponding to the five actual classes.

NEXT PHASE:
After local inference works, add a FastAPI backend and web upload UI.
