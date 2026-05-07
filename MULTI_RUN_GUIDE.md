# Multi-Run ASR Evaluation Guide

## 🎯 **New Output Structure for Multiple Runs**

Each run now creates **timestamped directories** to prevent overwriting:

```
results/
├── predictions/
│   ├── run_20231201_143022/
│   │   ├── whisper_original_auto.csv
│   │   ├── whisper_shortened_auto.csv
│   │   └── wav2vec2_*.csv
│   ├── run_20231201_151245/
│   │   └── ...
│   └── run_20231201_163000/
│       └── ...
├── comparisons/
│   ├── run_20231201_143022/
│   │   ├── asr_compare_20231201_143022.json
│   │   ├── run_metadata_20231201_143022.json
│   │   └── comprehensive_summary_20231201_143022.json
│   └── run_20231201_151245/
│       └── ...
└── multi_run_comparisons/
    ├── comparison_plots_20231201_170000.png
    ├── comparison_report_20231201_170000.json
    └── comparison_summary_20231201_170000.csv
```

## 🚀 **How to Run Multiple Evaluations**

### **Method 1: Different Parameters**

Run the main notebook multiple times with different CONFIG values:

```python
# Example configurations to test:
CONFIG = {
    "shorten_factor": 0.5,  # Test: 0.5, 0.75, 0.9
    "min_elongated_sec": 0.3,  # Test: 0.3, 0.5, 0.7
    "models_to_eval": ["whisper", "wav2vec2"],  # Test different model combos
}
```

### **Method 2: Different Data**

Test with different datasets:
- `data/test.csv` (original)
- `data/test_small.csv` (subset)
- Custom datasets

### **Method 3: Different Models**

Test different model combinations:
- `["whisper", "wav2vec2"]`
- `["whisper", "vosk"]`
- `["wav2vec2", "vosk"]`

## 📊 **Compare Results**

### **Using the Comparison Notebook**

1. Open `compare_multiple_runs_fixed.ipynb`
2. Run all cells to load previous runs
3. Customize which runs to compare:
   ```python
   # Compare all runs
   selected_runs = run_dirs
   
   # Compare 3 most recent
   selected_runs = run_dirs[:3]
   
   # Compare specific runs
   selected_runs = [run_dirs[0], run_dirs[2]]
   ```

### **Comparison Features**

- **Performance Trends**: See how models improve over runs
- **Parameter Impact**: Analyze effect of `shorten_factor` and `min_elongated_sec`
- **Robustness Analysis**: Compare model stability across conditions
- **Visual Comparisons**: Comprehensive plots and charts

## 🎛️ **Experiment Ideas**

### **Experiment 1: Optimize Shorten Factor**
```python
# Run 1: Conservative shortening
CONFIG["shorten_factor"] = 0.9

# Run 2: Moderate shortening  
CONFIG["shorten_factor"] = 0.75

# Run 3: Aggressive shortening
CONFIG["shorten_factor"] = 0.5
```

### **Experiment 2: Optimize Voice Detection**
```python
# Run 1: Sensitive detection
CONFIG["min_elongated_sec"] = 0.3

# Run 2: Standard detection
CONFIG["min_elongated_sec"] = 0.5

# Run 3: Conservative detection
CONFIG["min_elongated_sec"] = 0.7
```

### **Experiment 3: Model Comparison**
```python
# Run 1: All models
CONFIG["models_to_eval"] = ["whisper", "wav2vec2", "vosk"]

# Run 2: Transformer models only
CONFIG["models_to_eval"] = ["whisper", "wav2vec2"]

# Run 3: Lightweight models only
CONFIG["models_to_eval"] = ["vosk"]
```

## 📈 **What You Can Learn**

- **Optimal Parameters**: Which settings give best CER
- **Model Robustness**: Which models handle audio changes best
- **Parameter Sensitivity**: How configuration changes affect performance
- **Performance Trade-offs**: Speed vs accuracy considerations

## 🔧 **Troubleshooting**

### **No Previous Runs Found**
- Make sure you've run the main evaluation notebook at least once
- Check that `results/comparisons/` directory exists

### **Missing Data for Some Runs**
- Each run needs its metadata file: `run_metadata_YYYYMMDD_HHMMSS.json`
- Check that all files were generated successfully

### **Memory Issues**
- Compare fewer runs at once: `selected_runs = run_dirs[:2]`
- Use CPU instead of GPU for comparison notebook

---

**🎯 Ready to run multiple experiments and compare results!**
