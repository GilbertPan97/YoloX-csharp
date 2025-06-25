using System;
using System.IO;
using System.Linq;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

public class YamlLoader
{
    public static RootConfig Load(string yamlPath)
    {
        var yamlText = File.ReadAllText(yamlPath);
        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build();
        return deserializer.Deserialize<RootConfig>(yamlText);
    }

    public static void Print(YoloModelConfig config)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("=============== YOLO Model Configuration (YAML) ===============");

        Console.WriteLine("📌 Model Info:");
        Console.WriteLine($"  - Name: {config.name}");
        Console.WriteLine($"  - Version: {config.version}");
        Console.WriteLine($"  - Type: {config.type}");
        Console.WriteLine($"  - Task: {config.task}\n");

        Console.WriteLine("📥 Input:");
        Console.WriteLine($"  - Shape: [{string.Join(", ", config.input.shape)}]");
        Console.WriteLine($"  - Shape Sequence: ({string.Join(", ", config.input.shape_seq)})");
        Console.WriteLine($"  - Dtype: {config.input.dtype}");
        Console.WriteLine($"  - Normalize: {config.input.normalize}");
        Console.WriteLine($"  - Pad & Resize: {config.input.pad_resize}");
        Console.WriteLine($"  - Format: {config.input.format}\n");

        Console.WriteLine("📤 Outputs:");
        foreach (var output in config.output)
        {
            Console.WriteLine($"  - Output: {output.name} (Index {output.output_index})");
            Console.WriteLine($"    - Shape: [{string.Join(", ", output.shape)}]");
            Console.WriteLine($"    - Shape Sequence: ({string.Join(", ", output.shape_seq)})");
            Console.WriteLine($"    - Type: {output.type}");
            if (output.format != null)
            {
                if (output.format.box_range != null)
                    Console.WriteLine($"    - Box Range: [{string.Join(", ", output.format.box_range)}]");
                Console.WriteLine($"    - Objectness: {output.format.objectness}");
                if (output.format.class_range != null)
                    Console.WriteLine($"    - Class Range: [{string.Join(", ", output.format.class_range)}]");
                if (output.format.mask_coeff_range != null)
                    Console.WriteLine($"    - Mask Coeff Range: [{string.Join(", ", output.format.mask_coeff_range)}]");
            }
            Console.WriteLine();
        }

        Console.WriteLine("📐 Anchor/Grid Info:");
        Console.WriteLine($"  - Anchors: {config.anchors}");
        Console.WriteLine($"  - Grids: {string.Join(" + ", config.grids.Select(g => $"{g[0]}x{g[1]}"))}");
        Console.WriteLine($"  - Classes: {config.num_classes}");
        Console.WriteLine("==================== End of Configuration ====================");
        Console.ResetColor();
    }

    public static bool ValidateCfgStructure(YoloModelConfig cfg)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("🔍 Validating YOLO Model Configuration......");

        bool isValid = true;

        // 👉 1. Validate prediction count based on anchor/grid configuration
        Console.Write("🔧 Validating prediction count... ");
        int expectedPredictions = 0;

        // FIXME: Not finish
        // Compute expected prediction count
        if (cfg.type == "anchor-based")
        {
            expectedPredictions = cfg.grids.Sum(g => g[0] * g[1] * cfg.anchors);
        }
        else
        {
            expectedPredictions = cfg.grids.Sum(g => g[0] * g[1]);
        }

        // Find the detection output entry
        var detOutput = cfg.output.FirstOrDefault(o => o.type == "detection");
        if (detOutput == null)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("❌ Missing detection output.");
            return false;
        }

        if (detOutput.shape.Count < 3)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("❌ Detection output shape should have at least 3 dimensions.");
            return false;
        }

        int actualPredictions = detOutput.shape[detOutput.shape_seq[1]];
        if (expectedPredictions != actualPredictions)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ Prediction count mismatch. Expected: {expectedPredictions}, Got: {actualPredictions}");
            isValid = false;
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✅ OK ({actualPredictions} predictions)");
        }

        // 👉 2. Validate format channel indices are within bounds
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write("🔧 Validating format index ranges... ");

        var fmt = detOutput.format;
        int ch = detOutput.shape[detOutput.shape_seq[2]];
        int maxIndex = -1;

        if (fmt == null)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("❌ Detection format is not defined.");
            return false;
        }

        // Check all index fields for upper bound
        if (fmt.box_range != null && fmt.box_range.Count > 0)
            maxIndex = Math.Max(maxIndex, fmt.box_range.Max());

        if (fmt.objectness >= 0)
            maxIndex = Math.Max(maxIndex, fmt.objectness);

        if (fmt.class_range != null && fmt.class_range.Count == 2)
            maxIndex = Math.Max(maxIndex, fmt.class_range[1]);

        if (fmt.mask_coeff_range != null && fmt.mask_coeff_range.Count == 2)
            maxIndex = Math.Max(maxIndex, fmt.mask_coeff_range[1]);

        if (maxIndex >= ch)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ Format index out of bounds. Max used: {maxIndex}, Channels available: {ch}");
            isValid = false;
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✅ OK (Max index: {maxIndex} < Channels: {ch})");
        }

        // Final status
        Console.ForegroundColor = isValid ? ConsoleColor.Green : ConsoleColor.Red;
        Console.WriteLine($"Validation {(isValid ? "Passed ✅" : "Failed ❌")}\n");
        Console.ResetColor();

        return isValid;
    }

}
