# ASR Project Script Structure

## 📁 **Current Clean Structure**

### **Core Scripts (Essential)**
```
scripts/
├── __init__.py
├── check_data.py                    # Data validation
├── shorten_elongated_voiced_segments.py  # Audio processing
├── apply_dictionary_correction.py   # Text post-processing
├── build_dictionary.py              # Dictionary utilities
├── convert_m4a_to_wav.py           # Audio format conversion
├── convert_to_mono.py               # Audio channel conversion
├── make_small_csv.py                # Dataset sampling
├── split_train_dev.py              # Dataset splitting
├── make_augmented_trainset.py       # Data augmentation
├── eval_whisper_lora.py             # LoRA model evaluation
├── train_whisper_lora.py            # LoRA training
├── train_whisper_lora_small.py     # Small LoRA training
├── train_whisper_lora_specaug.py    # LoRA training with SpecAug
└── compare_3_asr_original_vs_shortened.ipynb  # Legacy comparison notebook
```

### **Organized Modules (New Structure)**
```
scripts/
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── audio_utils.py              # Audio loading/processing
│   ├── text_utils.py               # Text normalization
│   └── timing_utils.py              # Execution timing
├── models/                          # ASR model wrappers
│   ├── __init__.py
│   └── asr_models.py               # Whisper, wav2vec2, Vosk
├── audio_processing/                # Audio processing
│   ├── __init__.py
│   └── voice_segmentation.py       # Voice segment detection
└── evaluation/                      # Evaluation utilities
    ├── __init__.py
    ├── evaluate_models.py           # Sequential evaluation
    └── parallel_evaluate_models.py  # Parallel evaluation
```

## 🗑️ **Removed Scripts**

### **Obsolete Evaluation Scripts**
- ❌ `baseline_wav2vec_beam_eval.py` - Replaced by modular evaluation
- ❌ `baseline_wav2vec_eval.py` - Replaced by modular evaluation  
- ❌ `baseline_wav2vec_lm_eval.py` - Replaced by modular evaluation
- ❌ `baseline_whisper_eval.py` - Replaced by modular evaluation
- ❌ `compare_asr_models_full.py` - Replaced by notebook system
- ❌ `compare_asr_original_vs_shortened.py` - Replaced by notebook system
- ❌ `compare_original_vs_shortened.py` - Replaced by notebook system
- ❌ `compare_3_asr_original_vs_shortened.py` - Replaced by notebook system
- ❌ `compare_two_prediction_sets.py` - Integrated into evaluation modules
- ❌ `error_type_analysis.py` - Integrated into evaluation modules

### **Utility Scripts**
- ❌ `test_imports.py` - Development/testing script
- ❌ `make_pitch_test.py` - Empty/unused script
- ❌ `.DS_Store` - macOS system file

## 🎯 **What to Use Now**

### **For Evaluation:**
- **Main Notebook**: `asr_robustness_evaluation.ipynb`
- **Comparison Notebook**: `compare_multiple_runs_with_timing.ipynb`
- **Benchmark Notebook**: `benchmark_parallel_configurations.ipynb`
- **Module**: `scripts/evaluation/parallel_evaluate_models.py`

### **For Audio Processing:**
- **Module**: `scripts/audio_processing/voice_segmentation.py`
- **Script**: `scripts/shorten_elongated_voiced_segments.py` (legacy)

### **For Model Operations:**
- **Module**: `scripts/models/asr_models.py`
- **Utilities**: `scripts/utils/audio_utils.py`, `scripts/utils/text_utils.py`

### **For Training:**
- **LoRA Training**: `scripts/train_whisper_lora*.py`
- **LoRA Evaluation**: `scripts/eval_whisper_lora.py`

### **For Data Prep:**
- **Validation**: `scripts/check_data.py`
- **Conversion**: `scripts/convert_*.py`
- **Augmentation**: `scripts/make_augmented_trainset.py`

## 📊 **Benefits of Cleanup**

### **Reduced Complexity**
- **Before**: 25+ scattered scripts
- **After**: 12 essential scripts + 4 organized modules

### **Improved Organization**
- **Modular Design**: Related functions grouped together
- **Clear Separation**: Utilities, models, processing, evaluation
- **Reusable Components**: Easy to import and use

### **Better Maintenance**
- **Single Source**: One place for each functionality
- **Consistent API**: Standardized interfaces
- **Documentation**: Clear purpose for each module

### **Enhanced Features**
- **Parallel Processing**: Configurable threading
- **Timing System**: Comprehensive execution tracking
- **Notebook Integration**: Jupyter-friendly workflows
- **Multi-run Support**: Timestamped results

---

**🎯 Clean, organized, and ready for production use!**
