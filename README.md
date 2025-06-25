# YOLO-ONNXRuntime Inference in C#

This repository provides a C# implementation for performing inference with YOLO models using ONNX Runtime. It supports configurable model input/output structures using YAML and currently supports YOLOv7 and YOLOv9.

## 🔧 Requirements

- .NET 6.0 or later
- OpenCvSharp
- onnxruntime (via NuGet)
- YamlDotNet (via NuGet)

## 🚀 Run Inference

```bash
dotnet run -- \
  --model=./models/yolov7-seg.onnx \
  --imgDir=./imgs \
  --labels=./models/labels_algae.txt \
  --saveDir=./runs \
  --yaml=./cfg/yolov7-seg.yaml
```

### Parameters

| Argument     | Description                              | Required | Default                             |
|--------------|------------------------------------------|----------|-------------------------------------|
| `--model`    | Path to the ONNX model file              | No       | `../../../models/yolov7-seg.onnx`   |
| `--imgDir`   | Directory containing input images        | No       | `../../../imgs`                     |
| `--labels`   | Path to label text file                  | No       | `../../../models/labels_algae.txt`  |
| `--saveDir`  | Directory to save results                | No       | `../../../runs`                     |
| `--yaml`     | Path to YOLO YAML configuration file     | Yes      |                                     |

## 📦 Features

- 🔁 Batch inference over image directory
- 📋 YAML-based input/output model configuration
- 🧠 Support for YOLOv7, YOLOv9
- 🎥 Video recording of inference results
- 🖼️ Visual result rendering with bounding boxes and segmentation masks

## 📄 Output

- Inference images saved in `--saveDir`
- MP4 video saved as `--saveDir/video/inference_result.mp4`

## 🙏 Acknowledgement

This project only performs model inference and does **not** include any model training logic. All model weights used in this repository are exported from the original training repositories:

- **YOLOv7**: https://github.com/WongKinYiu/yolov7  
- **YOLOv9**: https://github.com/WongKinYiu/yolov9  

Please refer to the above repositories for model training code, dataset preparation, and export scripts. All rights and credits belong to the original authors. This repo does not modify or redistribute training code.

## 🆘 Help

Run with help flag:

```bash
dotnet run -- --help
```

## 📁 Notes

- The YAML file must follow `yolov-model-interface` format.
- Frame size is currently fixed to 5472x3648 for video rendering.
