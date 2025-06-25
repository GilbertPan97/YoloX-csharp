using System.Collections.Generic;

public class RootConfig
{
    public MetaConfig meta { get; set; }
    public YoloModelConfig model { get; set; }
}

public class MetaConfig
{
    public string config_type { get; set; }
    public string description { get; set; }
    public string format_version { get; set; }
}

public class YoloModelConfig
    {
        public string name { get; set; }
        public string version { get; set; }
        public string type { get; set; }
        public string task { get; set; }
        public InputConfig input { get; set; }
        public List<OutputConfig> output { get; set; }
        public int anchors { get; set; }
        public List<List<int>> grids { get; set; }
        public int num_classes { get; set; }
    }

public class InputConfig
{
    public List<int> shape { get; set; }
    public List<int> shape_seq { get; set; }
    public string dtype { get; set; }
    public bool normalize { get; set; }
    public bool pad_resize { get; set; }
    public string format { get; set; }
}

public class OutputConfig
{
    public string name { get; set; }
    public int output_index { get; set; }
    public List<int> shape { get; set; }
    public List<int> shape_seq { get; set; }
    public string type { get; set; }
    public OutputFormat format { get; set; }
}

public class OutputFormat
{
    public List<int> box_range { get; set; }
    public int objectness { get; set; }
    public List<int> class_range { get; set; }
    public List<int> mask_coeff_range { get; set; }
    public int channels { get; set; }
    public List<int> spatial { get; set; }
}
