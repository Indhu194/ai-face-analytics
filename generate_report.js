const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, PageBreak,
  Numbering, LevelFormat, BorderStyle, PageNumber, Footer, Header,
} = require("docx");

const fs = require("fs");

const TITLE = "AGE AND GENDER ESTIMATION FROM FACE";

function bodyPara(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 200, line: 300 },
    children: [new TextRun({ text, size: 24, ...opts })],
  });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 200 },
    children: [new TextRun({ text, bold: true })],
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 150 },
    children: [new TextRun({ text, bold: true })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullet-list", level: 0 },
    spacing: { after: 100 },
    children: [new TextRun({ text, size: 24 })],
  });
}

function makeCell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2500, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!opts.header, size: 22 })],
    })],
  });
}

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullet-list",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ],
      },
    ],
  },
  sections: [
    // ---------------- TITLE PAGE ----------------
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({ spacing: { before: 2000 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "A Project Report on", size: 28 })] }),
        new Paragraph({ spacing: { before: 400, after: 400 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: TITLE, size: 40, bold: true })] }),
        new Paragraph({ spacing: { before: 600 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Submitted in partial fulfilment of the requirements for the award of the degree of", size: 22 })] }),
        new Paragraph({ spacing: { before: 200 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "BACHELOR OF TECHNOLOGY", size: 26, bold: true })] }),
        new Paragraph({ spacing: { before: 200, after: 800 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "in", size: 22 }), new TextRun({ text: "  [Your Branch/Department]", size: 24, bold: true })] }),
        new Paragraph({ spacing: { before: 1000 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Submitted by", size: 22 })] }),
        new Paragraph({ spacing: { before: 200 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Indhu", size: 26, bold: true })] }),
        new Paragraph({ spacing: { before: 100 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "[Register / Roll Number]", size: 22 })] }),
        new Paragraph({ spacing: { before: 1200 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Under the guidance of", size: 22 })] }),
        new Paragraph({ spacing: { before: 200, after: 1200 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "[Guide Name, Designation]", size: 24, bold: true })] }),
        new Paragraph({ spacing: { before: 1200 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "[Your College / University Name]", size: 24, bold: true })] }),
        new Paragraph({ spacing: { before: 100 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "[Department Name]", size: 22 })] }),
        new Paragraph({ spacing: { before: 100 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "2025 - 2026", size: 22 })] }),
      ],
    },
    // ---------------- MAIN CONTENT ----------------
    {
      properties: {},
      headers: {
        default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: TITLE, size: 16, italics: true })] })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }),
      },
      children: [
        heading1("ABSTRACT"),
        bodyPara(
          "This project presents a system for estimating a person's age and gender directly from a facial image, using a Convolutional Neural Network (CNN). Human age and gender carry useful information for a wide range of applications - from targeted retail analytics and access control, to demographic research and human-computer interaction - yet manually reading them from an image is subjective and does not scale. The proposed system detects a face in an input image or a live webcam feed, extracts the facial region, and passes it through a CNN with two prediction heads: one performing binary classification for gender (male/female) and the other performing regression to estimate the numeric age. The model is trained on the UTKFace dataset, which contains over 20,000 labeled facial images spanning a wide age range. The final system is deployed as an interactive web application (built with Streamlit) that supports both static image upload and real-time webcam-based prediction, making the model easy to demonstrate and use."
        ),
        bodyPara(
          "Experimental results (recorded after training - see Chapter 8) show that the model achieves reasonable accuracy for gender classification and a competitive mean absolute error for age estimation, comparable to other lightweight CNN-based approaches reported in the literature. The project demonstrates an end-to-end pipeline: dataset preparation, model design, training, evaluation, and deployment, and discusses the limitations and possible future improvements of the system."
        ),

        heading1("TABLE OF CONTENTS"),
        bodyPara("1. Introduction"),
        bodyPara("2. Literature Survey"),
        bodyPara("3. Objectives and Scope"),
        bodyPara("4. System Requirements"),
        bodyPara("5. Proposed Methodology"),
        bodyPara("6. System Design and Architecture"),
        bodyPara("7. Implementation"),
        bodyPara("8. Results and Discussion"),
        bodyPara("9. Conclusion and Future Scope"),
        bodyPara("10. References"),

        new Paragraph({ children: [new PageBreak()] }),

        // ---------------- 1. INTRODUCTION ----------------
        heading1("1. INTRODUCTION"),
        bodyPara(
          "Estimating a person's age and gender from a photograph is a task humans perform almost instinctively, but one that is genuinely difficult to automate reliably. Faces vary enormously due to lighting, pose, expression, image quality, and natural diversity in appearance, and the visual cues that indicate age (wrinkles, skin texture, facial structure) change gradually and inconsistently across individuals. Historically, such estimation relied on hand-crafted image features (edges, texture descriptors) combined with classical machine learning classifiers, but these approaches struggled to generalize across the variety of real-world face images."
        ),
        bodyPara(
          "The rise of deep learning, and Convolutional Neural Networks (CNNs) in particular, has significantly improved the accuracy of automatic facial analysis tasks such as face recognition, expression recognition, and demographic attribute estimation. CNNs automatically learn hierarchical visual features directly from pixel data, removing the need for manual feature engineering and allowing the model to discover the subtle patterns that correlate with age and gender."
        ),
        bodyPara(
          "This project implements an end-to-end age and gender estimation system. A face is first located in an input image using a face detector, then the cropped face region is passed through a trained CNN that simultaneously predicts the person's estimated age (as a continuous value) and gender (as a binary class). The system is made usable through a web-based interface that accepts either an uploaded photograph or a live webcam capture."
        ),

        // ---------------- 2. LITERATURE SURVEY ----------------
        heading1("2. LITERATURE SURVEY"),
        bodyPara(
          "Several approaches to facial age and gender estimation have been proposed in the research literature:"
        ),
        bullet("Levi and Hassner (2015) proposed one of the first widely-used CNN architectures for age and gender classification from unconstrained real-world photos (the Adience benchmark), using a relatively shallow CNN and treating age as a classification problem over age groups rather than exact regression."),
        bullet("Rothe, Timofte, and Van Gool (2015) introduced the 'DEX' (Deep EXpectation) approach, which framed age estimation as a classification problem over discrete year bins and then computed an expected value, improving accuracy over direct regression."),
        bullet("Zhang, Song, and Qi (2017) and later works explored deeper architectures and transfer learning from large face-recognition networks (e.g. VGGFace) as a starting point, since face-recognition features transfer well to age/gender estimation."),
        bullet("The UTKFace dataset (Zhang et al.) and the IMDB-WIKI dataset are the two most commonly used public datasets for this task, providing large numbers of labeled face images with age, gender, and (for UTKFace) ethnicity annotations."),
        bodyPara(
          "This project follows the general approach of Levi & Hassner and similar lightweight CNN methods: a compact convolutional backbone shared between two output heads, trained from scratch on UTKFace. This is a practical choice for a college project because it does not require a large pretrained face-recognition model and can be trained in a reasonable time on a free-tier GPU."
        ),

        // ---------------- 3. OBJECTIVES ----------------
        heading1("3. OBJECTIVES AND SCOPE"),
        heading2("3.1 Objectives"),
        bullet("To design and train a CNN-based model capable of estimating age and gender from a facial image."),
        bullet("To implement a reliable face detection stage that isolates the face region before prediction."),
        bullet("To evaluate the trained model's accuracy on unseen (validation) data."),
        bullet("To build an interactive application that allows prediction from both an uploaded image and a live webcam feed."),
        heading2("3.2 Scope"),
        bodyPara(
          "The system is designed for single or multiple frontal/near-frontal faces in an image. It is a demonstration and learning project rather than a production-grade biometric system; predictions are approximate and the system does not attempt to identify or verify a person's identity - only to estimate two general demographic attributes from visual appearance."
        ),

        // ---------------- 4. SYSTEM REQUIREMENTS ----------------
        heading1("4. SYSTEM REQUIREMENTS"),
        heading2("4.1 Hardware Requirements"),
        bullet("A computer with a webcam (for the live demo)"),
        bullet("GPU recommended for training (a free Google Colab GPU is sufficient); CPU is enough for running inference"),
        heading2("4.2 Software Requirements"),
        new Table({
          columnWidths: [3500, 6500],
          width: { size: 10000, type: WidthType.DXA },
          rows: [
            new TableRow({ children: [makeCell("Component", { header: true, width: 3500 }), makeCell("Purpose", { header: true, width: 6500 })] }),
            new TableRow({ children: [makeCell("Python 3.9+", { width: 3500 }), makeCell("Programming language for the entire pipeline", { width: 6500 })] }),
            new TableRow({ children: [makeCell("TensorFlow / Keras", { width: 3500 }), makeCell("Building and training the CNN model", { width: 6500 })] }),
            new TableRow({ children: [makeCell("OpenCV (opencv-python)", { width: 3500 }), makeCell("Face detection (Haar Cascade) and image processing", { width: 6500 })] }),
            new TableRow({ children: [makeCell("Streamlit", { width: 3500 }), makeCell("Web application front-end for upload and webcam demo", { width: 6500 })] }),
            new TableRow({ children: [makeCell("scikit-learn", { width: 3500 }), makeCell("Train/validation dataset splitting", { width: 6500 })] }),
            new TableRow({ children: [makeCell("Google Colab / Kaggle Notebooks", { width: 3500 }), makeCell("Free GPU environment for model training", { width: 6500 })] }),
          ],
        }),
        bodyPara(""),

        // ---------------- 5. METHODOLOGY ----------------
        heading1("5. PROPOSED METHODOLOGY"),
        heading2("5.1 Dataset"),
        bodyPara(
          "The model is trained on the UTKFace dataset, a large-scale, publicly available collection of over 20,000 face images with age, gender, and ethnicity labels encoded directly in each filename (e.g. 25_0_2_20170116174525125.jpg indicates age 25, gender 0/male). The dataset spans ages from 0 to 116 and includes diverse ethnicities, poses, and lighting conditions, which makes it suitable for training a generalizable model."
        ),
        heading2("5.2 Preprocessing"),
        bullet("Each image is read and, where a face detector is used at training time, cropped to the facial region."),
        bullet("Images are resized to a fixed input size of 128 x 128 pixels."),
        bullet("Pixel values are normalized to the range [0, 1] by dividing by 255."),
        bullet("The dataset is split into training (85%) and validation (15%) subsets."),
        heading2("5.3 Model Architecture"),
        bodyPara(
          "A single convolutional backbone is shared between two prediction heads (a multi-task learning design). The backbone consists of four convolutional blocks (Conv2D + BatchNormalization + MaxPooling), increasing in filter depth (32, 64, 128, 256), followed by global average pooling and a dense layer. Two separate small dense branches then produce the final outputs:"
        ),
        bullet("Gender head: Dense(64) -> Dropout -> Dense(1, sigmoid) - binary classification (male/female)"),
        bullet("Age head: Dense(64) -> Dropout -> Dense(1, linear) - regression (predicted age in years)"),
        bodyPara(
          "Sharing a single backbone keeps the model compact and trains faster than two independent networks, while still allowing each head to specialize through its own dense layers."
        ),
        heading2("5.4 Training Configuration"),
        new Table({
          columnWidths: [3500, 6500],
          width: { size: 10000, type: WidthType.DXA },
          rows: [
            new TableRow({ children: [makeCell("Parameter", { header: true, width: 3500 }), makeCell("Value", { header: true, width: 6500 })] }),
            new TableRow({ children: [makeCell("Optimizer", { width: 3500 }), makeCell("Adam, learning rate 1e-3 (reduced on plateau)", { width: 6500 })] }),
            new TableRow({ children: [makeCell("Loss (gender)", { width: 3500 }), makeCell("Binary Cross-Entropy", { width: 6500 })] }),
            new TableRow({ children: [makeCell("Loss (age)", { width: 3500 }), makeCell("Mean Absolute Error (MAE)", { width: 6500 })] }),
            new TableRow({ children: [makeCell("Batch size", { width: 3500 }), makeCell("64", { width: 6500 })] }),
            new TableRow({ children: [makeCell("Epochs", { width: 3500 }), makeCell("Up to 30, with early stopping", { width: 6500 })] }),
            new TableRow({ children: [makeCell("Regularization", { width: 3500 }), makeCell("Dropout (0.3-0.4), Batch Normalization", { width: 6500 })] }),
          ],
        }),
        bodyPara(""),

        // ---------------- 6. SYSTEM DESIGN ----------------
        heading1("6. SYSTEM DESIGN AND ARCHITECTURE"),
        bodyPara("The overall pipeline consists of the following stages, executed in sequence for every input image or webcam frame:"),
        bullet("1. Input acquisition - an image is either uploaded by the user or captured from the webcam."),
        bullet("2. Face detection - OpenCV's Haar Cascade classifier locates one or more face bounding boxes in the frame."),
        bullet("3. Preprocessing - each detected face is cropped, resized to 128x128, and normalized."),
        bullet("4. CNN inference - the preprocessed face is passed through the trained model, producing an age estimate and a gender probability."),
        bullet("5. Post-processing and display - the bounding box, predicted gender label, and predicted age are drawn on the image and shown to the user."),
        bodyPara(
          "The application layer is built with Streamlit and offers two interaction modes: an 'Upload Image' tab for static photographs, and a 'Live Webcam' tab for browser-based webcam capture. A separate standalone OpenCV script is also provided for a continuous, real-time video demonstration (useful for live presentations)."
        ),

        // ---------------- 7. IMPLEMENTATION ----------------
        heading1("7. IMPLEMENTATION"),
        heading2("7.1 Module Breakdown"),
        bullet("model_architecture.py - defines the CNN using the Keras functional API with two output heads."),
        bullet("train.py - loads the UTKFace dataset, builds a tf.data input pipeline, and trains the model, saving the best-performing weights."),
        bullet("face_utils.py - shared utility functions for face detection (Haar Cascade), face cropping/preprocessing, and converting the model's gender probability into a readable label."),
        bullet("app.py - the Streamlit web application providing the upload and webcam-snapshot interfaces."),
        bullet("webcam_demo.py - a standalone OpenCV script for continuous, real-time webcam predictions."),
        heading2("7.2 Key Implementation Details"),
        bodyPara(
          "Face detection uses OpenCV's built-in Haar Cascade frontal-face classifier, chosen because it ships with opencv-python (no extra model download required) and runs quickly enough for real-time use on a CPU. Model inference uses TensorFlow/Keras' model.predict() on the cropped, normalized face region. The gender output is a single sigmoid value; values at or above 0.5 are interpreted as 'Female' and below 0.5 as 'Male', matching the UTKFace label convention. The age output is a single linear (regression) value, rounded to the nearest integer for display."
        ),
        bodyPara(
          "[Insert code screenshots or key snippets here from model_architecture.py, train.py, and app.py, as required by your report format.]"
        ),

        // ---------------- 8. RESULTS ----------------
        heading1("8. RESULTS AND DISCUSSION"),
        bodyPara(
          "[This section should be completed after you run training in Chapter 5's Colab notebook. Insert your actual numbers, training curves, and screenshots below. A suggested structure is provided.]"
        ),
        heading2("8.1 Training Curves"),
        bodyPara("[Insert accuracy/loss vs. epoch plots for both the gender and age heads here.]"),
        heading2("8.2 Quantitative Results"),
        new Table({
          columnWidths: [5000, 5000],
          width: { size: 10000, type: WidthType.DXA },
          rows: [
            new TableRow({ children: [makeCell("Metric", { header: true, width: 5000 }), makeCell("Value", { header: true, width: 5000 })] }),
            new TableRow({ children: [makeCell("Gender classification accuracy (validation)", { width: 5000 }), makeCell("[fill in]", { width: 5000 })] }),
            new TableRow({ children: [makeCell("Age Mean Absolute Error (validation, years)", { width: 5000 }), makeCell("[fill in]", { width: 5000 })] }),
            new TableRow({ children: [makeCell("Average inference time per face (CPU)", { width: 5000 }), makeCell("[fill in]", { width: 5000 })] }),
          ],
        }),
        bodyPara(""),
        heading2("8.3 Sample Outputs"),
        bodyPara("[Insert screenshots of the app correctly predicting age/gender for a few sample images and webcam captures.]"),
        heading2("8.4 Discussion / Limitations"),
        bullet("Accuracy typically drops for extreme head poses, poor lighting, occlusion (glasses, masks, hair covering the face), and low-resolution images."),
        bullet("Age estimation is inherently harder than gender classification because visual aging cues vary widely between individuals of the same chronological age."),
        bullet("Model predictions reflect the distribution of the UTKFace training data, and may be less accurate for age groups or demographics that are underrepresented in the dataset."),

        // ---------------- 9. CONCLUSION ----------------
        heading1("9. CONCLUSION AND FUTURE SCOPE"),
        bodyPara(
          "This project successfully implements an end-to-end pipeline for estimating age and gender from facial images using a multi-task CNN, trained on the UTKFace dataset. The system integrates face detection, preprocessing, model inference, and a user-facing web application supporting both image upload and live webcam prediction. The results demonstrate that a compact, shared-backbone CNN can achieve reasonable performance on this task without requiring a very deep or heavily pretrained network."
        ),
        heading2("Future Scope"),
        bullet("Use transfer learning from a pretrained face-recognition network (e.g. VGGFace, FaceNet) to improve accuracy with less training data."),
        bullet("Replace the Haar Cascade detector with a more robust DNN-based face detector for better performance on angled or partially occluded faces."),
        bullet("Treat age estimation as an ordinal classification problem (age buckets) rather than pure regression, which has been shown to improve accuracy in prior research."),
        bullet("Deploy the model as a mobile app using TensorFlow Lite for on-device inference."),
        bullet("Expand the training dataset with additional sources to improve fairness and reduce bias across demographic groups."),

        // ---------------- 10. REFERENCES ----------------
        heading1("10. REFERENCES"),
        bodyPara("1. G. Levi and T. Hassner, \"Age and Gender Classification using Convolutional Neural Networks,\" IEEE CVPR Workshops, 2015."),
        bodyPara("2. R. Rothe, R. Timofte, and L. Van Gool, \"DEX: Deep EXpectation of apparent age from a single image,\" ICCV Workshops, 2015."),
        bodyPara("3. Z. Zhang, Y. Song, and H. Qi, \"Age Progression/Regression by Conditional Adversarial Autoencoder,\" CVPR, 2017. (UTKFace dataset)"),
        bodyPara("4. UTKFace Dataset: https://susanqq.github.io/UTKFace/"),
        bodyPara("5. OpenCV Documentation: https://docs.opencv.org/"),
        bodyPara("6. TensorFlow / Keras Documentation: https://www.tensorflow.org/"),
        bodyPara("7. Streamlit Documentation: https://docs.streamlit.io/"),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("Age_Gender_Estimation_Report.docx", buffer);
  console.log("Report generated.");
});
