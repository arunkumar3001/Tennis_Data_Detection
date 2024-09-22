# YOLOv8 Object Detection with Video Input 🎥🚀

This repository contains an **object detection project** using the **YOLOv8** model, trained on a custom dataset and implemented in **Google Colab**. The model processes video input to detect and annotate objects in real-time, and outputs a video with the detected objects highlighted.

## 🚀 Project Overview

- **Model**: YOLOv8 (You Only Look Once, version 8)
- **Framework**: Python, Google Colab
- **Libraries**: `ultralytics` (YOLOv8), `opencv-python`
- **Input**: Custom video dataset
- **Output**: Video with detected objects annotated (bounding boxes, labels)

## 📁 Dataset

The dataset used for training the YOLOv8 model contains custom-labeled videos/images. The dataset must be formatted in the YOLO format (images and corresponding `.txt` annotation files).

**Note**: Ensure your dataset is structured correctly with images in one folder and labels in another.

## 📦 Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/yolo-object-detection.git
    cd yolo-object-detection
    ```

2. **Google Colab Setup**:
    - Install the required dependencies:
      ```python
      !pip install ultralytics opencv-python
      ```

3. **Model Training**:
    - Load YOLOv8 and train the model using your custom dataset.
    ```python
    from ultralytics import YOLO

    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Nano model

    # Train the model on your dataset
    model.train(data='/content/drive/MyDrive/Your-Dataset/data.yaml', epochs=50, imgsz=640)
    ```

4. **Object Detection on Video**:
    - Load your trained model and perform object detection on a video.
    ```python
    import cv2
    from ultralytics import YOLO

    # Load trained model
    model = YOLO('/content/best_model.pt')

    # Process video input
    cap = cv2.VideoCapture('/content/drive/MyDrive/path_to_video.mp4')
    out = cv2.VideoWriter('/content/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        out.write(results[0].plot())

    cap.release()
    out.release()
    ```

## 🎯 Key Features

- **Real-Time Object Detection**: Detects objects in video frames and outputs an annotated video.
- **YOLOv8 Model**: Trained using a custom dataset, with flexible model size (nano, small, medium).
- **Video Output**: Detects objects from video input and saves results as a new annotated video.

## 🚀 Output Example

Once the video is processed, the output will contain bounding boxes and labels for each detected object in the frames.

## 🛠️ Tools & Technologies

- **YOLOv8**: State-of-the-art object detection model
- **Python**: Programming language
- **OpenCV**: For handling video input and output
- **Google Colab**: Platform for training the model

## 🔗 Links

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Google Colab](https://colab.research.google.com/)

## 💻 How to Use the Model

To use the trained model on a new video:
1. Train your model with the steps provided above.
2. Use the following code to test your model on a video file:
    ```python
    results = model('/content/path_to_new_video.mp4')
    ```

## 🎓 Learning Resources

- **YOLOv8 Documentation**: Learn about the YOLOv8 model and its capabilities.
- **OpenCV Documentation**: Official docs for working with video files in Python.

## 🤝 Contributions

Contributions, suggestions, and feedback are welcome! Please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.

---

🚀 Happy Coding! 🔍
