# Age and Gender Estimation from Face

Final-year B.Tech project: a CNN that detects a face in an image or webcam
feed and predicts **age** (regression) and **gender** (classification).

## Project Structure

```
age_gender_project/
├── model/
│   ├── model_architecture.py   # CNN definition (shared backbone + 2 heads)
│   ├── train.py                 # training script (run on Colab/Kaggle with GPU)
│   └── age_gender_model.h5      # (you generate this by training - not included)
├── utils/
│   └── face_utils.py            # face detection + preprocessing helpers
├── app/
│   ├── app.py                   # Streamlit web app (image upload + webcam snapshot)
│   └── webcam_demo.py           # OpenCV live/continuous webcam demo
├── requirements.txt
└── README.md
```

## Step 1 — Install dependencies (local machine)

```bash
pip install -r requirements.txt
```

## Step 2 — Train the model (Google Colab, free GPU)

You cannot train this efficiently on a normal laptop CPU, so use
[Google Colab](https://colab.research.google.com) (free GPU) or Kaggle Notebooks:

1. Open a new Colab notebook, set **Runtime → Change runtime type → GPU**.
2. Upload `model/model_architecture.py` and `model/train.py`.
3. Download the **UTKFace** dataset (~20,000 labeled face images, ~200MB):
   - Easiest: Kaggle dataset `jangedoo/utkface-new` (instructions are inside `train.py`)
   - Or manually from https://susanqq.github.io/UTKFace/
4. Edit `DATA_DIR` in `train.py` to point at the folder of images.
5. Run:
   ```bash
   python train.py
   ```
   This trains for up to 30 epochs (early-stopping included) and saves
   `age_gender_model.h5`.
6. Download `age_gender_model.h5` from Colab and place it inside your local
   `model/` folder.

**Expected results** (typical for this architecture on UTKFace): gender
accuracy in the ~88-92% range, mean absolute age error around 5-7 years.
Your exact numbers depend on training time/epochs — record them for your
report's Results section.

## Step 3 — Run the web app (image upload + webcam)

```bash
cd app
streamlit run app.py
```

This opens a browser tab with two tabs: **Upload Image** and **Live Webcam**
(the webcam tab takes a snapshot through the browser — simplest and most
reliable for a Streamlit app).

## Step 3b — Run the live, continuous webcam demo (for your presentation)

```bash
cd app
python webcam_demo.py
```

This opens your webcam directly via OpenCV and draws live predictions on
every frame — good for a real-time demo during your viva. Press `q` to quit.

## How it works (short version)

1. **Face detection**: OpenCV's Haar Cascade finds face bounding boxes in
   the frame (fast, built into `opencv-python`, no extra downloads).
2. **Preprocessing**: each detected face is cropped, resized to 128×128,
   and normalized to [0, 1].
3. **CNN prediction**: a shared convolutional backbone feeds two heads —
   a sigmoid output for gender (male/female) and a linear output for age
   (a single predicted number).
4. **Display**: bounding box + predicted age/gender are drawn on the image.

## Notes for your report / viva

- Dataset: UTKFace (~20,000 images, ages 0-116, labeled by filename).
- Why single shared CNN with two heads instead of two separate models:
  fewer parameters, faster training, and both tasks benefit from the same
  facial features (this is called *multi-task learning*).
- Losses: MAE (mean absolute error) for age regression, binary
  cross-entropy for gender classification, combined with loss weights.
- Limitations to mention: accuracy drops on poor lighting, extreme angles,
  occluded faces, and the model reflects biases present in the UTKFace
  dataset (age/ethnicity distribution isn't perfectly balanced).
- Possible future improvements: transfer learning from a pretrained face
  model (e.g. VGGFace, FaceNet embeddings), larger dataset, data
  augmentation, treating age as classification into buckets instead of
  regression.
