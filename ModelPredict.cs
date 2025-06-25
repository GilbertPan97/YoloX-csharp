using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class ModelPredict
{
    // Struct for input image info
    private struct InImgInfo
    {
        public bool DiffWithModel;
        public Size InImgSize;
    }

    // ORT handles
    private InferenceSession session_;
    private SessionOptions sessionOps_;
    private OrtMemoryInfo memoryInfo_;

    // Model info
    private List<string> inputNames_ = new List<string>();
    private List<string> outputNames_ = new List<string>();
    private int[] inputDims_ = Array.Empty<int>();
    private int[] outputDims_ = Array.Empty<int>();
    private YoloModelConfig cfg_;

    // Input image size
    private InImgInfo imgInfo_;

    // Inference results
    private List<float[]> bboxes_ = new List<float[]>();
    private List<float[]> minbboxes_ = new List<float[]>();
    private List<int> labels_ = new List<int>();
    private List<string> classesName_ = new List<string>();
    private List<float> scores_ = new List<float>();
    private List<Mat> masks_ = new List<Mat>();


    // ===================== Constructor & Destructor =====================//
    /// <summary>
    /// Initialize the ModelPredict instance.
    /// </summary>
    /// <param name="withGpu">If true, try to enable CUDA GPU acceleration; otherwise use CPU.</param>
    /// <param name="deviceId">CUDA device ID to use when GPU is enabled.</param>
    /// <param name="thread">Number of CPU threads for intra-op parallelism.</param>
    public ModelPredict(bool withGpu = false, int deviceId = 0, int thread = 1)
    {
        // Create session options
        sessionOps_ = new SessionOptions();

        // Set number of intra-op threads (parallelism for ops)
        sessionOps_.IntraOpNumThreads = thread;

        // Set graph optimization level to all
        sessionOps_.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        if (withGpu)
        {
            try
            {
                sessionOps_.AppendExecutionProvider_CUDA(deviceId);
                LogInfo(nameof(ModelPredict), $"Successfully enabled CUDA on GPU: {deviceId}");
            }
            catch (Exception ex)
            {
                LogWarning(nameof(ModelPredict), ex.Message);
                LogWarning(nameof(ModelPredict), "Failed to enable CUDA. The model will run on the CPU instead.");
                // throw if needed
            }
        }
        else
        {
            LogWarning(nameof(ModelPredict), "GPU option is not selected. The model will run on the CPU.");
        }

        memoryInfo_ = OrtMemoryInfo.DefaultInstance;
    }

    ~ModelPredict() { }


    // ===================== Public methods =====================//
    /// <summary>
    /// Loads an ONNX model from the specified path and sets the task type.
    /// </summary>
    /// <param name="modelPath">The path to the ONNX model file.</param>
    /// <param name="cfg"></param>
    /// <returns>True if the model was loaded successfully, false otherwise.</returns>
    public bool LoadModel(string modelPath, YoloModelConfig cfg)
    {
        string func = nameof(LoadModel);
        LogInfo(func, "Start loading model.");

        try
        {
            session_ = new InferenceSession(modelPath, sessionOps_);
        }
        catch (OnnxRuntimeException ex)
        {
            LogError(func, $"Failed to load model: {ex.Message}");
            return false;
        }

        inputNames_.Clear();
        foreach (var input in session_.InputMetadata)
        {
            inputNames_.Add(input.Key);
            LogInfo(func, $"Model input name: {input.Key}");
        }

        outputNames_.Clear();
        foreach (var output in session_.OutputMetadata)
        {
            outputNames_.Add(output.Key);
            LogInfo(func, $"Model output name: {output.Key}");
        }

        // Retrive model configuration
        if (CheckYoloCFG(cfg, session_))
        {
            inputDims_ = session_.InputMetadata.ElementAt(0).Value.Dimensions.ToArray();
            outputDims_ = session_.OutputMetadata.ElementAt(0).Value.Dimensions.ToArray();
            cfg_ = cfg;
        }
        else
            return false;

        WarmUpModel();
        LogInfo(func, "Successfully loaded model.");

        return true;
    }

    /// <summary>
    /// Loads a list of class label names used for predictions.
    /// </summary>
    /// <param name="classes">A list of class label strings.</param>
    /// <returns>True if labels are successfully loaded, false otherwise.</returns>
    public bool LabelCategories(List<string> classes)
    {
        string func = nameof(LabelCategories);
        if (classes == null || classes.Count == 0)
        {
            LogError(func, "Label list is null or empty.");
            return false;
        }

        // Preserve empty lines, just trim each entry
        classesName_ = classes.Select(c => c.Trim()).ToList();

        LogInfo(func, $"Loaded {classesName_.Count} class labels (including empty lines).");
        return true;
    }

    /// <summary>
    /// Runs inference on the given image and saves prediction results including bounding boxes, labels, scores, and masks (if applicable).
    /// </summary>
    /// <param name="inputImg">The input image (OpenCvSharp Mat) to run inference on.</param>
    /// <param name="scoreThresh">Score threshold to filter low-confidence detections. Default is 0.7.</param>
    /// <returns>True if inference runs successfully, false otherwise.</returns>
    public bool PredictAction(Mat inputImg, float scoreThresh = 0.7f)
    {
        string func = nameof(PredictAction);

        // Clear previous results
        bboxes_.Clear();
        labels_.Clear();
        scores_.Clear();
        masks_.Clear();

        int modelInHeight = cfg_.input.shape[2];
        int modelInWidth = cfg_.input.shape[3];

        Mat inferImg = inputImg.Clone();
        if (modelInHeight != inputImg.Rows || modelInWidth != inputImg.Cols)
        {
            Size inSize = new Size(modelInWidth, modelInHeight);

            if (cfg_.input.pad_resize)
                inferImg = Letterbox(inferImg, inSize);
            else
                Cv2.Resize(inferImg, inferImg, inSize);

            imgInfo_.DiffWithModel = true;
            imgInfo_.InImgSize = new Size(inputImg.Cols, inputImg.Rows);
        }
        else
        {
            imgInfo_.DiffWithModel = false;
            imgInfo_.InImgSize = new Size(inputImg.Cols, inputImg.Rows);
        }

        // Prepare tensor
        var tensorValue = new List<float>();
        var inputTensor = CreateTensor(inferImg, cfg_.input.shape.ToArray(), tensorValue, cfg_.input.format, cfg_.input.normalize);
        string[] inputNames = inputNames_.ToArray();
        string[] outputNames = outputNames_.ToArray();

        // Inference
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputTensors;
        var sw = Stopwatch.StartNew();
        try
        {
            outputTensors = session_.Run(
                new[] { NamedOnnxValue.CreateFromTensor<float>(inputNames[0], inputTensor) },
                outputNames
            );
        }
        catch (OnnxRuntimeException e)
        {
            LogError(func, e.Message);
            return false;
        }
        sw.Stop();
        LogInfo(func, "Inference time consume : " + sw.Elapsed.TotalSeconds + " s.");

        // Prepare output data
        var outputData = new List<(float[], long[])>();
        foreach (var output in outputTensors)
        {
            var tensor = output.AsTensor<float>();
            float[] data = tensor.ToArray();
            long[] dims = tensor.Dimensions.ToArray().Select(d => (long)d).ToArray();
            outputData.Add((data, dims));
        }

        // Extract detect data
        float[] pred = outputData[0].Item1;
        long[] shapePred = outputData[0].Item2;
        int nc = cfg_.num_classes;

        var detOutput = cfg_.output.FirstOrDefault(o => o.name == "det");
        int[] det_seq = detOutput.shape_seq.ToArray();
        var det_fmt = detOutput.format;
        var allBatchDetections = NonMaxSuppression(ReorderPredTo012(pred, shapePred, det_seq), ReorderShapeFast(shapePred, det_seq), det_fmt);

        // FIXME: Only support one batch inference
        foreach (var det in allBatchDetections)
        {
            if (det.Count == 0) continue;

            foreach (var detection in det)
            {
                var box = new float[] { detection[0], detection[1], detection[2], detection[3] };
                float conf = detection[4];
                int cls = (int)detection[5];

                bboxes_.Add(box);
                scores_.Add(conf);
                labels_.Add(cls);

                if (cfg_.task == "InstanceSeg" )
                {
                    var maskOutput = cfg_.output.FirstOrDefault(o => o.name == "mask");
                    float[] predMasksProto = outputData[maskOutput.output_index].Item1;
                    long[] shapePredMasksProto = outputData[maskOutput.output_index].Item2;
                    var masksCoeff = detection.Skip(6).ToList();
                    var instanceMask = ComputeInstanceMask(predMasksProto, shapePredMasksProto, masksCoeff, inferImg.Size());
                    var cleanedMask = ApplyBoxMaskConstraint(instanceMask, box);
                    masks_.Add(cleanedMask);

                    // // DEBUG: View cleaned mask and infer image
                    // Cv2.ImShow("Cleaned Mask", cleanedMask);
                    // Cv2.WaitKey(10);
                    // Cv2.ImShow("Infer image", inferImg);
                    // Cv2.WaitKey(0);

                    // Cv2.DestroyAllWindows();
                }
            }

            RescaleCoords(inferImg.Size(), bboxes_, inputImg.Size());

            if (masks_.Count > 0)
            {
                RecoverMasksToOriginalSize(masks_, inputImg.Size(), new Size(modelInWidth, modelInHeight));
            }
        }

        return true;
    }

    /// <summary>
    /// Gets the list of predicted bounding boxes from the last inference.
    /// </summary>
    /// <returns>A list of bounding boxes represented by float arrays [x1, y1, x2, y2].</returns>
    public List<float[]> GetBoundingBoxes() => bboxes_;

    /// <summary>
    /// Get the minimum bounding boxes (rotated rectangles) calculated from masks.
    /// If not already calculated, will compute them.
    /// </summary>
    /// <returns>List of float arrays representing 4 points of rotated rectangles (x0,y0,x1,y1,x2,y2,x3,y3)</returns>
    public List<float[]> GetMinBoundingBoxes()
    {
        if (minbboxes_.Count != masks_.Count)
        {
            CalculateMinBoundingBoxes();
        }
        return minbboxes_;
    }

    /// <summary>
    /// Gets the list of instance masks predicted in the last inference.
    /// </summary>
    /// <returns>A list of OpenCvSharp Mat objects representing binary masks.</returns>
    public List<Mat> GetPredictMasks() => masks_;

    /// <summary>
    /// Gets the list of class labels predicted in the last inference.
    /// </summary>
    /// <returns>A list of integer class indices corresponding to detected objects.</returns>
    public List<int> GetPredictLabels() => labels_;

    /// <summary>
    /// Gets the list of confidence scores for each detection from the last inference.
    /// </summary>
    /// <returns>A list of float values representing prediction confidences.</returns>
    public List<float> GetPredictScores() => scores_;


    // ===================== Private methods =====================//
    /// <summary>
    /// Validates whether the ONNX model matches the given YOLO model configuration.
    /// </summary>
    /// <param name="cfg">YOLO model configuration, including expected input/output shapes and indices.</param>
    /// <param name="session">ONNX inference session from the loaded model.</param>
    /// <returns>True if the model structure matches the configuration; otherwise, false.</returns>
    private bool CheckYoloCFG(YoloModelConfig cfg, InferenceSession session)
    {
        const string funcName = "CheckYoloCFG";

        // 1. Validate input metadata
        var modelInputMeta = session.InputMetadata;
        if (modelInputMeta.Count != 1)
        {
            LogError(funcName, "❌ Model should have exactly one input tensor.");
            return false;
        }

        var modelInput = modelInputMeta.First();
        var actualInputShape = modelInput.Value.Dimensions;

        if (!cfg.input.shape.SequenceEqual(actualInputShape))
        {
            LogError(funcName, $"❌ Input shape mismatch: Config [{string.Join(", ", cfg.input.shape)}] ≠ Model [{string.Join(", ", actualInputShape)}]");
            return false;
        }

        // 2. Validate number of outputs
        // var modelOutputs = session.OutputMetadata;
        // if (cfg.output.Count != modelOutputs.Count)
        // {
        //     LogError(funcName, $"❌ Output count mismatch: Config {cfg.output.Count} ≠ Model {modelOutputs.Count}");
        //     return false;
        // }

        //3. Check only outputs specified in cfg by output_index
        var modelOutputMeta = session.OutputMetadata;
        foreach (var outputCfg in cfg.output)
        {
            if (outputCfg.output_index < 0 || outputCfg.output_index >= modelOutputMeta.Count)
            {
                LogError(funcName, $"❌ Invalid output index {outputCfg.output_index} in config (model has {modelOutputMeta.Count} outputs).");
                return false;
            }

            var actualOutputShape = modelOutputMeta.ElementAt(outputCfg.output_index).Value.Dimensions;
            if (!outputCfg.shape.SequenceEqual(actualOutputShape))
            {
                LogError(funcName, $"❌ Output shape mismatch at index {outputCfg.output_index}:\n  Config: [{string.Join(", ", outputCfg.shape)}]\n  Model:  [{string.Join(", ", actualOutputShape)}]");
                return false;
            }
        }

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] [{funcName}] ✅ Model and config match.");
        Console.ResetColor();

        return true;
    }


    /// <summary>
    /// Performs a warm-up run on the ONNX model to initialize internal states and optimize performance.
    /// It creates dummy input tensors filled with constant values to execute one inference pass.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown if input or output names are not set, or if the warm-up inference fails.
    /// </exception>
    private void WarmUpModel()
    {
        if (inputNames_ == null || inputNames_.Count == 0)
            throw new InvalidOperationException("Input names are not set.");
        if (outputNames_ == null || outputNames_.Count == 0)
            throw new InvalidOperationException("Output names are not set.");

        var dummyInputs = new List<NamedOnnxValue>();

        foreach (var inputName in inputNames_)
        {
            var inputMeta = session_.InputMetadata[inputName];
            var inputDims = inputMeta.Dimensions.ToArray();

            // Handle dynamic dimensions (-1) by replacing with 1
            for (int i = 0; i < inputDims.Length; i++)
            {
                if (inputDims[i] < 0) inputDims[i] = 1;
            }

            int numElements = inputDims.Aggregate(1, (a, b) => a * b);
            float[] dummyData = Enumerable.Repeat(0.3f, numElements).ToArray();

            var tensor = new DenseTensor<float>(dummyData, inputDims);
            dummyInputs.Add(NamedOnnxValue.CreateFromTensor(inputName, tensor));
        }

        try
        {
            // Run dummy inference
            using var results = session_.Run(dummyInputs);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("Model warm-up failed: " + ex.Message);
        }
    }

    /// <summary>
    /// Resizes and pads an image to fit into a specified shape while maintaining the aspect ratio.
    /// This is commonly used to prepare images for input into neural networks.
    /// </summary>
    /// <param name="img">The input image to be resized and padded.</param>
    /// <param name="newShape">The target size to fit the image into.</param>
    /// <param name="color">Optional padding color (default is gray [114,114,114]).</param>
    /// <param name="autoResize">If true, adjusts padding to be multiples of the stride.</param>
    /// <param name="scaleFill">If true, stretches the image to fill the new shape without preserving aspect ratio.</param>
    /// <param name="scaleUp">If false, prevents upscaling the image beyond its original size.</param>
    /// <param name="stride">The stride size used for auto padding alignment (default 32).</param>
    /// <returns>The resized and padded image as a new Mat object.</returns>
    public static Mat Letterbox(Mat img, Size newShape, Scalar? color = null,
                        bool autoResize = false, bool scaleFill = false,
                        bool scaleUp = true, int stride = 32)
    {
        Scalar padColor = color ?? new Scalar(114, 114, 114);
        Size shape = new Size(img.Width, img.Height); // original size

        // Compute scale ratio
        float r = Math.Min((float)newShape.Height / shape.Height, (float)newShape.Width / shape.Width);
        if (!scaleUp)
            r = Math.Min(r, 1.0f);

        // Compute new size without padding
        Size newUnpad = new Size((int)Math.Round(shape.Width * r), (int)Math.Round(shape.Height * r));

        // Compute padding
        float dw = newShape.Width - newUnpad.Width;
        float dh = newShape.Height - newUnpad.Height;

        if (autoResize)
        {
            dw %= stride;
            dh %= stride;
        }
        else if (scaleFill)
        {
            dw = 0;
            dh = 0;
            newUnpad = newShape;
            r = (float)newShape.Width / shape.Width;
        }

        dw /= 2.0f;
        dh /= 2.0f;

        // Resize
        Mat resizedImg = new Mat();
        if (shape != newUnpad)
            Cv2.Resize(img, resizedImg, newUnpad, 0, 0, InterpolationFlags.Linear);
        else
            resizedImg = img;

        // Pad
        int top = (int)Math.Round(dh - 0.1f);
        int bottom = (int)Math.Round(dh + 0.1f);
        int left = (int)Math.Round(dw - 0.1f);
        int right = (int)Math.Round(dw + 0.1f);

        Mat paddedImg = new Mat();
        Cv2.CopyMakeBorder(resizedImg, paddedImg, top, bottom, left, right, BorderTypes.Constant, padColor);

        return paddedImg;
    }

    /// <summary>
    /// Creates a tensor from an OpenCV Mat image, converting to the desired shape and format.
    /// Supports BGR->RGB conversion and normalization to [0,1].
    /// </summary>
    /// <param name="mat">Input OpenCV Mat image (assumed BGR format)</param>
    /// <param name="dims">Target tensor dimensions in NCHW format (e.g. [1, 3, height, width])</param>
    /// <param name="valueBuffer">Buffer to hold the tensor data (cleared and filled inside)</param>
    /// <param name="format">Format string: "CHW" or others (default HWC)</param>
    /// <returns>Tensor of float matching the dims shape</returns>
    /// <exception cref="Exception">Throws if dims length, batch size, or channels mismatch</exception>
    private static Tensor<float> CreateTensor(Mat mat, int[] dims, List<float> valueBuffer, string format, bool normalize)
    {
        int rows = mat.Rows;
        int cols = mat.Cols;
        int channels = mat.Channels();

        Mat matRef = new Mat();
        if (mat.Type() != MatType.CV_32FC(channels))
        {
            mat.ConvertTo(matRef, MatType.CV_32FC(channels));
        }
        else
        {
            matRef = mat;
        }

        // BGR to RGB conversion
        Mat rgbMat = new Mat();
        Cv2.CvtColor(matRef, rgbMat, ColorConversionCodes.BGR2RGB);

        // Normalize to [0,1]
        Mat targetMat = new Mat();
        rgbMat.ConvertTo(targetMat, MatType.CV_32FC3, normalize ? 1.0 / 255.0 : 1.0);

        if (dims.Length != 4)
            throw new Exception("dims mismatch.");
        if (dims[0] != 1)
            throw new Exception("batch != 1");

        int targetChannels = dims[1];
        int targetHeight = dims[2];
        int targetWidth = dims[3];
        int targetTensorSize = targetChannels * targetHeight * targetWidth;

        if (targetChannels != channels)
            throw new Exception("channel mismatch!");

        // Resize image to target size if needed
        Mat resizeMat = new Mat();
        if (targetHeight != rows || targetWidth != cols)
        {
            Cv2.Resize(targetMat, resizeMat, new Size(targetWidth, targetHeight));
        }
        else
        {
            resizeMat = targetMat;
        }

        // Clear and prepare buffer
        valueBuffer.Clear();
        valueBuffer.Capacity = targetTensorSize;

        if (format == "CHW")
        {
            // Split channels
            Mat[] matChannels = Cv2.Split(resizeMat);

            for (int i = 0; i < channels; i++)
            {
                float[] channelData = new float[targetHeight * targetWidth];
                Marshal.Copy(matChannels[i].Data, channelData, 0, channelData.Length);
                valueBuffer.AddRange(channelData);
                matChannels[i].Dispose();
            }
        }
        else
        {
            // HWC format: just copy raw data
            float[] rawData = new float[targetTensorSize];
            Marshal.Copy(resizeMat.Data, rawData, 0, targetTensorSize);
            valueBuffer.AddRange(rawData);
        }

        // Create and return tensor directly
        return new DenseTensor<float>(valueBuffer.ToArray(), dims);
    }

    /// <summary>
    /// Reorder a 3D shape according to the semantic axis mapping.
    /// Returns shape in [batch, boxes, boxElements] order.
    /// </summary>
    /// <param name="shapePred">Original shape, typically 3D like [1, 73, 25200]</param>
    /// <param name="seq">Semantic axis order (e.g., [0,2,1]) where 0=batch, 1=boxes, 2=elements</param>
    /// <returns>Reordered int[3] array in standard [batch, boxes, boxElements] order</returns>
    public static long[] ReorderShapeFast(long[] shapePred, int[] seq)
    {
        // Avoid bounds check at runtime
        if ((uint)shapePred.Length != 3 || (uint)seq.Length != 3)
            throw new ArgumentException("Expected shapePred and seq to be of length 3.");

        // Fast access without LINQ
        long dim0 = shapePred[0];
        long dim1 = shapePred[1];
        long dim2 = shapePred[2];

        // Map semantic axis index to value: result[i] = shape[seq.IndexOf(i)]
        // So result[0] = batch size, result[1] = box count, result[2] = element count
        long[] result = new long[3];
        for (int i = 0; i < 3; ++i)
        {
            int sourceIndex = seq[0] == i ? 0 :
                            seq[1] == i ? 1 :
                            seq[2] == i ? 2 : -1;

            // Inline index check
            if ((uint)sourceIndex > 2)
                throw new ArgumentException($"Invalid axis {i} in seq.");

            result[i] = sourceIndex == 0 ? dim0 :
                        sourceIndex == 1 ? dim1 :
                        dim2;
        }

        return result;
    }

    /// <summary>
    /// Reorders a flat float[] tensor from an arbitrary 3D axis order to standard [batch, boxes, elements].
    /// </summary>
    /// <param name="input">Flat input data.</param>
    /// <param name="shape">Original shape of input data.</param>
    /// <param name="seq">Axis meaning mapping. E.g. [0,2,1] means input shape is [batch, elem, boxes].</param>
    /// <returns>Reordered float[] with shape [batch, boxes, elements].</returns>
    public static float[] ReorderPredTo012(float[] input, long[] shape, int[] seq)
    {
        if (shape.Length != 3 || seq.Length != 3)
            throw new ArgumentException("Shape and seq must be of length 3.");

        // Convert shape to int
        int[] dims = shape.Select(s => (int)s).ToArray();

        // Get index of each semantic axis (0=batch, 1=box, 2=elem)
        int axisB = Array.IndexOf(seq, 0);  // where batch is
        int axisX = Array.IndexOf(seq, 1);  // where box   is
        int axisC = Array.IndexOf(seq, 2);  // where elem  is

        int B = dims[axisB];
        int X = dims[axisX];
        int C = dims[axisC];

        float[] output = new float[B * X * C];

        // Prepare strides for original shape
        int[] strides = new int[3];
        strides[2] = 1;
        strides[1] = dims[2];
        strides[0] = dims[1] * dims[2];

        for (int b = 0; b < B; b++)
        {
            for (int x = 0; x < X; x++)
            {
                for (int c = 0; c < C; c++)
                {
                    // Original index in input
                    int[] idx = new int[3];
                    idx[axisB] = b;
                    idx[axisX] = x;
                    idx[axisC] = c;

                    int srcIndex = idx[0] * strides[0] + idx[1] * strides[1] + idx[2];
                    int dstIndex = (b * X + x) * C + c;

                    output[dstIndex] = input[srcIndex];
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Performs Non-Maximum Suppression (NMS) on model predictions to filter overlapping bounding boxes.
    /// </summary>
    /// <param name="pred">Flattened prediction array containing bounding boxes, scores, and optional mask coefficients.</param>
    /// <param name="shapePred">Shape of the prediction tensor, typically [batch, numBoxes, boxElements].</param>
    /// <param name="fmt">Output format definition, including index ranges for box, objectness, class scores, and optional masks.</param>
    /// <param name="confThresh">Confidence threshold to filter low-confidence boxes.</param>
    /// <param name="iouThresh">IoU threshold to suppress overlapping boxes.</param>
    /// <param name="maxDet">Maximum number of detections to keep per image.</param>
    /// <returns>
    /// A list of detections per batch. Each detection is a list of floats: 
    /// [x1, y1, x2, y2, confidence, class_id, ...optional mask coefficients].
    /// </returns>
    private static List<List<List<float>>> NonMaxSuppression(
        float[] pred, long[] shapePred, OutputFormat fmt,
        float confThresh = 0.25f, float iouThresh = 0.45f, int maxDet = 300)
    {
        int batchSize = (int)shapePred[0];
        int numBoxes = (int)shapePred[1];
        int boxElements = (int)shapePred[2];
        int predSize = numBoxes * boxElements;

        var output = new List<List<List<float>>>(batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            var itemsPred = new List<List<float>>();

            for (int i = 0; i < numBoxes; i++)
            {
                int offset = b * predSize + i * boxElements;

                // Check objectness score
                if (fmt.objectness != 0)
                {
                    float confidence = pred[offset + fmt.objectness];
                    if (confidence < confThresh)
                        continue;
                }

                // Check classes score
                var classScores = new List<float>();
                int minClassIdx = fmt.class_range.Min();
                int maxClassIdx = fmt.class_range.Max();
                for (int j = minClassIdx; j < maxClassIdx + 1; j++)
                    classScores.Add(pred[offset + j]);

                float maxScore = classScores.Max();
                int classIdx = classScores.IndexOf(maxScore);

                // if (classIdx >= 5)
                //     throw new Exception($"Detected class ID {classIdx} is not in the allowed class list.");

                if (maxScore < confThresh)
                    continue;

                // Retrive bbox result
                var item = new List<float>
                {
                    pred[offset + fmt.box_range.Min()],     // x
                    pred[offset + fmt.box_range.Min() + 1], // y
                    pred[offset + fmt.box_range.Min() + 2], // w
                    pred[offset + fmt.box_range.Min() + 3]  // h
                };

                XYWH2XYXY(item);

                // Retrive class information
                item.Add(maxScore); // class confidence
                item.Add(classIdx); // class index

                // Retrive extra coefficients (such as mask coefficients)
                var extraCoeff = new List<float>();
                int minCoefIdx = fmt.mask_coeff_range.Min();
                int maxCoefIdx = fmt.mask_coeff_range.Max();
                for (int j = minCoefIdx; j < maxCoefIdx + 1; j++)
                    extraCoeff.Add(pred[offset + j]);

                if (extraCoeff.Count > 0)
                    item.AddRange(extraCoeff);

                itemsPred.Add(item);
            }

            // Sort by confidence descending
            itemsPred.Sort((a, b) => b[4].CompareTo(a[4]));

            var keep = new List<List<float>>();
            while (itemsPred.Count > 0)
            {
                var current = itemsPred[0];
                keep.Add(current);
                itemsPred.RemoveAt(0);

                itemsPred.RemoveAll(box => IoU(current, box) > iouThresh);

                if (keep.Count >= maxDet)
                    break;
            }

            output.Add(keep);
        }

        return output;
    }

    /// <summary>
    /// Convert bounding box from [center_x, center_y, width, height] format
    /// to [x_min, y_min, x_max, y_max] format.
    /// </summary>
    /// <param name="box">List of floats representing a box in XYWH format.</param>
    private static void XYWH2XYXY(List<float> box)
    {
        float x = box[0], y = box[1], w = box[2], h = box[3];
        box[0] = x - w / 2; // x_min
        box[1] = y - h / 2; // y_min
        box[2] = x + w / 2; // x_max
        box[3] = y + h / 2; // y_max
    }

    /// <summary>
    /// Calculate Intersection over Union (IoU) between two bounding boxes in XYXY format.
    /// </summary>
    /// <param name="box1">First bounding box [x_min, y_min, x_max, y_max]</param>
    /// <param name="box2">Second bounding box [x_min, y_min, x_max, y_max]</param>
    /// <returns>IoU value between 0 and 1.</returns>
    private static float IoU(List<float> box1, List<float> box2)
    {
        float x1 = Math.Max(box1[0], box2[0]);
        float y1 = Math.Max(box1[1], box2[1]);
        float x2 = Math.Min(box1[2], box2[2]);
        float y2 = Math.Min(box1[3], box2[3]);

        float interArea = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

        return interArea / (area1 + area2 - interArea + 1e-6f);
    }

    /// <summary>
    /// Compute an instance segmentation mask using prototype masks and mask coefficients.
    /// </summary>
    /// <param name="masks">Flattened array of prototype masks (channels x height x width).</param>
    /// <param name="shape">Shape array: [batch, channels, height, width].</param>
    /// <param name="maskCoeff">Mask coefficients for linear combination.</param>
    /// <param name="inferSize">Target output size (width, height).</param>
    /// <param name="threshold">Threshold for binarizing mask.</param>
    /// <param name="applyMorph">Whether to apply morphological open and close operations.</param>
    /// <returns>Binary mask as OpenCV Mat.</returns>
    public static Mat ComputeInstanceMask(float[] masks, long[] shape, List<float> maskCoeff, Size inferSize, float threshold = 0.5f, bool applyMorph = true)
    {
        int c = (int)shape[1]; // channels (proto count)
        int h = (int)shape[2];
        int w = (int)shape[3];

        // Step 1: Linear combination
        Mat mask = new Mat(h, w, MatType.CV_32F, Scalar.All(0));
        float[] temp = new float[h * w];

        for (int i = 0; i < c; ++i)
        {
            Array.Copy(masks, i * h * w, temp, 0, h * w);

            Mat proto = new Mat(h, w, MatType.CV_32F);
            proto.SetArray<float>(temp); // 显式指定类型参数

            Cv2.Add(mask, proto * maskCoeff[i], mask); // mask += proto * coeff
        }

        // Step 2: Sigmoid activation: 1 / (1 + exp(-x))
        Mat negMask = new Mat();
        Cv2.Multiply(mask, -1.0, negMask);
        Mat expMask = new Mat();
        Cv2.Exp(negMask, expMask);

        Mat activated = new Mat();
        Cv2.Add(expMask, 1.0, expMask); // expMask = exp(-x) + 1
        Cv2.Divide(1.0, expMask, activated); // activated = 1 / (1 + exp(-x))

        // Step 3: Resize
        Mat resized = new Mat();
        Cv2.Resize(activated, resized, inferSize, 0, 0, InterpolationFlags.Linear);

        // Step 4: Threshold
        Mat binary = new Mat();
        Cv2.Threshold(resized, binary, threshold, 1.0, ThresholdTypes.Binary);

        // Step 5: Morphology (optional)
        if (applyMorph)
        {
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
            Cv2.MorphologyEx(binary, binary, MorphTypes.Open, kernel);
            Cv2.MorphologyEx(binary, binary, MorphTypes.Close, kernel);
        }

        return binary;
    }

    /// <summary>
    /// Applies a bounding box constraint to the given mask.
    /// The output mask retains values only within the bounding box region,
    /// while all other areas are set to zero.
    /// </summary>
    /// <param name="mask">Input mask as a single-channel float matrix (CV_32F).</param>
    /// <param name="boxXYXY">Bounding box coordinates in [x_min, y_min, x_max, y_max] format.</param>
    /// <returns>A new mask constrained within the bounding box.</returns>
    /// <exception cref="ArgumentException">Thrown if input mask type is not CV_32F.</exception>
    private static Mat ApplyBoxMaskConstraint(Mat mask, float[] boxXYXY)
    {
        if (mask.Type() != MatType.CV_32F)
            throw new ArgumentException("Input mask must be CV_32F type.");

        Mat constrained = Mat.Zeros(mask.Size(), MatType.CV_32F);

        int x1 = Math.Clamp((int)Math.Floor(boxXYXY[0]), 0, mask.Cols - 1);
        int y1 = Math.Clamp((int)Math.Floor(boxXYXY[1]), 0, mask.Rows - 1);
        int x2 = Math.Clamp((int)Math.Ceiling(boxXYXY[2]), 0, mask.Cols - 1);
        int y2 = Math.Clamp((int)Math.Ceiling(boxXYXY[3]), 0, mask.Rows - 1);

        Rect roi = new Rect(x1, y1, Math.Max(1, x2 - x1), Math.Max(1, y2 - y1));

        if (roi.X >= 0 && roi.Y >= 0 && roi.X + roi.Width <= mask.Cols && roi.Y + roi.Height <= mask.Rows)
        {
            mask[roi].CopyTo(constrained[roi]);
        }

        return constrained;
    }

    /// <summary>
    /// Rescales bounding box coordinates from a model input size back to the original image size.
    /// Also removes any padding added during preprocessing and clips coordinates to image boundaries.
    /// </summary>
    /// <param name="img1Shape">Size of the model input image (width, height).</param>
    /// <param name="coords">List of bounding boxes, each box is a float array [x_min, y_min, x_max, y_max].</param>
    /// <param name="img0Shape">Size of the original image (width, height).</param>
    /// <param name="ratioPad">
    /// Optional parameter containing precomputed scaling and padding values:
    /// ratioPad[0][0] = scale factor (gain),
    /// ratioPad[1][0] = padding in X,
    /// ratioPad[1][1] = padding in Y.
    /// If not provided, the function computes these values automatically.
    /// </param>
    private static void RescaleCoords(Size img1Shape, List<float[]> coords, Size img0Shape, List<List<float>> ratioPad = null)
    {
        float gain, padX, padY;

        // If ratioPad is not provided, calculate gain and padding
        if (ratioPad == null || ratioPad.Count == 0)
        {
            gain = Math.Min((float)img1Shape.Height / img0Shape.Height, (float)img1Shape.Width / img0Shape.Width);
            padX = (img1Shape.Width - img0Shape.Width * gain) / 2.0f;
            padY = (img1Shape.Height - img0Shape.Height * gain) / 2.0f;
        }
        else
        {
            gain = ratioPad[0][0];
            padX = ratioPad[1][0];
            padY = ratioPad[1][1];
        }

        foreach (var box in coords)
        {
            // Remove padding
            box[0] -= padX;
            box[1] -= padY;
            box[2] -= padX;
            box[3] -= padY;

            // Scale to original image size
            box[0] /= gain;
            box[1] /= gain;
            box[2] /= gain;
            box[3] /= gain;

            // Clip coordinates to image boundaries
            box[0] = Math.Max(0f, Math.Min(box[0], img0Shape.Width - 1));
            box[1] = Math.Max(0f, Math.Min(box[1], img0Shape.Height - 1));
            box[2] = Math.Max(0f, Math.Min(box[2], img0Shape.Width - 1));
            box[3] = Math.Max(0f, Math.Min(box[3], img0Shape.Height - 1));
        }
    }

    /// <summary>
    /// Recovers instance masks from the model input size to the original image size.
    /// This reverses the effect of letterbox resizing and padding applied during model preprocessing.
    /// </summary>
    /// <param name="masks">List of masks as CV_32F Mats with model input size.</param>
    /// <param name="oriImgSize">Original image size (width, height).</param>
    /// <param name="modelInputSize">
    /// Model input size (width, height).
    /// If default(Size) is passed, behavior is undefined (model input size must be provided).
    /// </param>
    /// <exception cref="Exception">Thrown if mask type or dimensions do not match expected model input format.</exception>
    private static void RecoverMasksToOriginalSize(List<Mat> masks, Size oriImgSize, Size modelInputSize = default)
    {
        int modelW = modelInputSize.Width;
        int modelH = modelInputSize.Height;
        int oriW = oriImgSize.Width;
        int oriH = oriImgSize.Height;

        // Calculate scaling factor and padding
        float scale = Math.Min((float)modelW / oriW, (float)modelH / oriH);
        int newW = (int)Math.Round(oriW * scale);
        int newH = (int)Math.Round(oriH * scale);
        int padX = (modelW - newW) / 2;
        int padY = (modelH - newH) / 2;

        // Define crop region of interest (ROI)
        Rect roi = new Rect(padX, padY, newW, newH);

        for (int i = 0; i < masks.Count; i++)
        {
            var mask = masks[i];

            // Ensure mask type is float32 and size is correct
            if (mask.Type() != MatType.CV_32F)
                throw new Exception("Mask must be of type CV_32F");

            if (mask.Rows != modelH || mask.Cols != modelW)
                throw new Exception("Mask dimensions must match model input size");

            // Crop the valid region (remove padding)
            Mat cropped = new Mat(mask, roi);

            // Resize the cropped mask to original image size
            Mat resized = new Mat();
            Cv2.Resize(cropped, resized, oriImgSize, 0, 0, InterpolationFlags.Linear);

            masks[i] = resized; // Overwrite original mask
        }
    }

    /// <summary>
    /// Calculate minimum area bounding boxes (rotated rectangles) from masks_
    /// and store the results in minbboxes_ as float arrays of 8 elements (4 points).
    /// </summary>
    private void CalculateMinBoundingBoxes()
    {
        minbboxes_.Clear();

        for (int i = 0; i < masks_.Count; i++)
        {
            Mat mask = masks_[i];

            // Convert mask to 8-bit single channel if needed
            if (mask.Type() != MatType.CV_8U)
            {
                Mat mask8U = new Mat();
                mask.ConvertTo(mask8U, MatType.CV_8U, 255.0);
                mask = mask8U;
            }

            // Find contours from the mask
            Point[][] contours;
            HierarchyIndex[] hierarchy;
            Cv2.FindContours(mask, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            if (contours.Length == 0)
            {
                // No contour found, add an empty box
                minbboxes_.Add(new float[8] { 0, 0, 0, 0, 0, 0, 0, 0 });
                continue;
            }

            // Find the largest contour by area
            int maxContourIdx = 0;
            double maxArea = 0;
            for (int c = 0; c < contours.Length; c++)
            {
                double area = Cv2.ContourArea(contours[c]);
                if (area > maxArea)
                {
                    maxArea = area;
                    maxContourIdx = c;
                }
            }

            // Get the minimum area rotated rectangle for the largest contour
            RotatedRect minRect = Cv2.MinAreaRect(contours[maxContourIdx]);

            // Extract the 4 vertices of the rectangle
            Point2f[] pts = minRect.Points();

            float[] boxPts = new float[8];
            for (int j = 0; j < 4; j++)
            {
                boxPts[j * 2] = pts[j].X;
                boxPts[j * 2 + 1] = pts[j].Y;
            }

            minbboxes_.Add(boxPts);
        }
    }

    private void LogInfo(string func, string msg) =>
        Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] [{func}] INFO: {msg}");

    private void LogWarning(string func, string msg)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] [{func}] WARNING: {msg}");
        Console.ResetColor();
    }

    private void LogError(string func, string msg)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.Error.WriteLine($"[{DateTime.Now:HH:mm:ss}] [{func}] ERROR: {msg}");
        Console.ResetColor();
    }
}
