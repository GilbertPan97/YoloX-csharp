using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using OpenCvSharp;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using System.Data.SqlTypes;

class Program
{
    // Define color codes for terminal output
    const string RED = "\x1b[31m";
    const string GREEN = "\x1b[32m";
    const string YELLOW = "\x1b[33m";
    const string RESET = "\x1b[0m";

    static void Main(string[] args)
    {
        if (args.Length == 0 || args.Any(a => a == "--help" || a == "-h"))
        {
            PrintUsage();
            return;
        }

        var argDict = args.Select(arg => arg.Split('='))
                        .Where(parts => parts.Length == 2)
                        .ToDictionary(parts => parts[0], parts => parts[1]);

        string modelPath = argDict.GetValueOrDefault("--model", "../../../models/yolov9-seg.onnx");
        string imgDir = argDict.GetValueOrDefault("--imgDir", "../../../imgs");
        string categoriesPath = argDict.GetValueOrDefault("--labels", "../../../models/labels_algae.txt");
        string saveDir = argDict.GetValueOrDefault("--saveDir", "../../../runs");
        string configPath = argDict.GetValueOrDefault("--yaml", "../../../cfg/yolov9-seg.yaml");

        // Check required arguments
        if (string.IsNullOrEmpty(modelPath) || string.IsNullOrEmpty(imgDir) ||
            string.IsNullOrEmpty(categoriesPath) || string.IsNullOrEmpty(saveDir) ||
            string.IsNullOrEmpty(configPath))
        {
            Console.WriteLine($"{RED}ERROR:{RESET} Missing required arguments.\n");
            PrintUsage();
            return;
        }

        Console.WriteLine($"{GREEN}INFO:{RESET} modelPath = {modelPath}");
        Console.WriteLine($"{GREEN}INFO:{RESET} imgDir = {imgDir}");
        Console.WriteLine($"{GREEN}INFO:{RESET} labels = {categoriesPath}");
        Console.WriteLine($"{GREEN}INFO:{RESET} saveDir = {saveDir}");

        string videoDir = Path.Combine(saveDir, "video");
        string videoPath = Path.Combine(videoDir, "inference_result.mp4");

        // Ensure output directories exist
        Directory.CreateDirectory(saveDir);
        Directory.CreateDirectory(videoDir);

        // Load image file names
        List<string> imgPaths, imgNames;
        ReadFileNamesInDir(imgDir, out imgPaths, out imgNames);

        // Load YOLO yaml configuration
        var root = YamlLoader.Load(configPath);
        if (root.meta?.config_type != "yolov-model-interface")
        {
            Console.WriteLine("❌ YAML is not a valid YOLO interface config.");
            return;
        }

        Console.WriteLine("✅ YAML config loaded successfully.\n");
        YamlLoader.Print(root.model);
        YamlLoader.ValidateCfgStructure(root.model);

        // Create ModelPredict and load model
        var modelPredict = new ModelPredict(withGpu: true, deviceId: 0, thread: 1);

        bool loaded = modelPredict.LoadModel(modelPath, root.model);
        if (!loaded)
        {
            Console.WriteLine("ERROR: Model load failed.");
            return;
        }

        List<string> categories = LoadCategories(categoriesPath);
        modelPredict.LabelCategories(categories);

        var infRender = new Renderer(categories);

        Console.WriteLine($"INFO: All inference images: {imgPaths.Count}");

        // MP4 settings
        int frameWidth = 5472;
        int frameHeight = 3648;
        int fps = 1;                // 100ms per frame
        int fourcc = VideoWriter.FourCC('a', 'v', 'c', '1');
        using var videoWriter = new VideoWriter(videoPath, fourcc, fps, new Size(frameWidth, frameHeight));

        for (int i = 0; i < imgPaths.Count; i++)
        {
            Console.WriteLine($"INFO: inference at: {i}, img name is: {imgNames[i]}");

            Mat img = Cv2.ImRead(imgPaths[i]);
            float scoreThresh = 0.4f;

            bool status = modelPredict.PredictAction(img, scoreThresh);
            infRender.SetImage(img);

            Mat resultImg = infRender.RenderInference(0.6f,
                modelPredict.GetBoundingBoxes(),
                modelPredict.GetPredictMasks(),
                modelPredict.GetPredictLabels(),
                modelPredict.GetPredictScores());

            // Resize for video
            Cv2.Resize(resultImg, resultImg, new Size(frameWidth, frameHeight));

            // Display
            string winName = "Inference result";
            Cv2.NamedWindow(winName, WindowFlags.Normal);
            Cv2.ImShow(winName, resultImg);
            Cv2.WaitKey(10);

            // Save image
            string savePath = Path.Combine(saveDir, imgNames[i]);
            Cv2.ImWrite(savePath, resultImg);

            // Write video frame
            videoWriter.Write(resultImg);
        }

        Console.WriteLine($"Inference done. MP4 video saved to: {videoPath}");
    }

    static void PrintUsage()
    {
        Console.WriteLine($"{YELLOW}Usage:{RESET} dotnet run -- --model=xxx.onnx --imgDir=path --labels=path --saveDir=path");
        Console.WriteLine($"{YELLOW}Required arguments:{RESET}");
        Console.WriteLine($"  {GREEN}--model=FILE{RESET}       Path to the ONNX model file");
        Console.WriteLine($"  {GREEN}--imgDir=DIR{RESET}       Directory containing input images");
        Console.WriteLine($"  {GREEN}--labels=FILE{RESET}      Path to class label text file");
        Console.WriteLine($"  {GREEN}--saveDir=DIR{RESET}      Directory to save output results");
        Console.WriteLine($"  {GREEN}--yaml=FILE{RESET}        PATH to yolo model description file");
        Console.WriteLine($"{YELLOW}Optional:{RESET}");
        Console.WriteLine($"  {GREEN}--help / -h{RESET}        Show this help message");
    }

    static void ReadFileNamesInDir(string dirPath, out List<string> fullPaths, out List<string> fileNames)
    {
        fullPaths = new List<string>();
        fileNames = new List<string>();

        if (!Directory.Exists(dirPath))
            return;

        var files = Directory.GetFiles(dirPath);
        Array.Sort(files); // Optional sort

        foreach (var file in files)
        {
            fullPaths.Add(file);
            fileNames.Add(Path.GetFileName(file));
        }
    }

    static List<string> LoadCategories(string filePath)
    {
        var categories = new List<string>();

        if (!File.Exists(filePath))
        {
            Console.WriteLine("Failed to open file: " + filePath);
            return categories;
        }

        foreach (var line in File.ReadLines(filePath))
        {
            categories.Add(line.Trim());
        }

        return categories;
    }
}
