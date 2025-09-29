
# Fish Image Classification Project

## Overview

This project is a **Deep Learning-based Fish Species Classifier** that can identify different types of fish from images. The system uses pre-trained convolutional neural network (CNN) models such as **MobileNet, VGG16, ResNet50, InceptionV3, and EfficientNetB0** and fine-tunes them for classification. The project also includes a **Streamlit web app** for interactive predictions.

---

## Features

* Classifies **11 different fish species** including trout, shrimp, sea bass, and more.
* Uses **pre-trained CNN models** with transfer learning for faster and accurate training.
* Implements **data augmentation** to improve model generalization.
* Provides **confidence scores** for each prediction.
* Includes a **web-based interface** for users to upload an image and see predictions instantly.

---

## Dataset

* The dataset contains images of **11 fish species**.
* Directory structure:

```
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

---

## Project Structure

```
fish-image-classification/
├── data/                       # Dataset folders (train/val/test)
├── models/                     # Saved trained models
│   ├── vgg16_trained_model.h5
│   ├── resnet50_trained_model.h5
│   └── ...
├── streamlit_app.py             # Streamlit application
├── train_models.py              # Script to train all models
├── evaluate_models.py           # Evaluate models on test data
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-link>
cd fish-image-classification
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

3. Install additional packages if needed:

```bash
pip install tensorflow keras streamlit numpy pandas matplotlib seaborn pillow scikit-learn
```

---

## Training

* Open `train_models.py` to train any of the CNN models.
* Example: Training VGG16

```python
python train_models.py --model VGG16
```

* Training uses **data augmentation** and **transfer learning** to improve accuracy.
* Models are saved in the `models/` folder after training.

---

## Evaluation

* Run `evaluate_models.py` to evaluate models on the test dataset.
* The script computes:

  * **Accuracy**
  * **Precision**
  * **Recall**
  * **F1-Score**
  * **Confusion Matrix**
* Results are visualized using **matplotlib** and **seaborn**.

---

## Streamlit Web App

* Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

* Upload a fish image to predict its species.
* Displays **predicted class**, **confidence score**, and a **confidence chart for all classes**.

---

## Results

* Model accuracies on the test dataset:

  * **VGG16:** 94%
  * **ResNet50:** 96%
  * **MobileNet:** 95%
  * **InceptionV3:** 93%
  * **EfficientNetB0:** 97%
* **EfficientNetB0** achieved the highest overall performance.

---

## Future Improvements

* Expand dataset to include more fish species.
* Use **real-time webcam input** for live predictions.
* Deploy the app on **cloud services** for wider access.
* Experiment with **ensemble models** to further improve accuracy.

