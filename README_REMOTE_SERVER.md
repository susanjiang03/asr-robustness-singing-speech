# ASR Robustness Evaluation - Remote Server Guide

## Overview
This project evaluates ASR model robustness on Chinese opera singing by comparing performance between original and audio-modified versions.

## Quick Start for Remote Servers

### 1. Upload Your Data

#### Required Files:
- **Audio files**: Upload to `data/clips/test_fixed/` directory
- **CSV dataset**: Upload as `data/test.csv` with format:
  ```csv
  audio,text
  data/clips/test_fixed/opera_001.wav,海岛冰轮初转腾
  data/clips/test_fixed/opera_002.wav,众将士听我把令行
  ```
- **Vosk model** (optional): Upload to `models/vosk-model-small-cn-0.22/`

#### Directory Structure:
```
data/
├── clips/
│   └── test_fixed/
│       ├── opera_001.wav
│       └── ...
├── test.csv
└── dev.csv (optional)

models/
└── vosk-model-small-cn-0.22/ (optional)
```

### 2. Install Dependencies

```bash
pip install torch torchaudio transformers librosa soundfile jiwer opencc-python-reimplemented vosk tqdm pandas matplotlib seaborn
```

### 3. Run the Evaluation

Open `asr_robustness_evaluation.ipynb` and execute cells in order:

1. **Environment Setup** - Installs dependencies and imports modules
2. **Configuration** - Update paths in the CONFIG section
3. **Data Verification** - Checks uploaded files
4. **Audio Processing** - Creates shortened versions
5. **Model Evaluation** - Runs ASR models on both datasets
6. **Results Analysis** - Generates visualizations and analysis
7. **Export Results** - Packages results for download

### 4. Configuration

Update the CONFIG section in the notebook:

```python
CONFIG = {
    "original_csv": "data/test.csv",
    "output_audio_dir": "data/clips/test_shortened",
    "shortened_csv": "data/test_shortened.csv",
    "vosk_model_path": "models/vosk-model-small-cn-0.22",
    "models_to_eval": ["whisper", "wav2vec2", "vosk"],  # Remove "vosk" if not available
    # ... other settings
}
```

### 5. Expected Outputs

#### Results Structure:
```
results/
├── predictions/
│   ├── whisper_original_auto.csv
│   ├── whisper_shortened_auto.csv
│   ├── wav2vec2_original_auto.csv
│   └── ...
└── comparisons/
    ├── asr_compare_YYYYMMDD_HHMMSS.json
    ├── asr_compare_YYYYMMDD_HHMMSS.csv
    ├── robustness_analysis_YYYYMMDD_HHMMSS.csv
    └── asr_comparison_plots_YYYYMMDD_HHMMSS.png
```

#### Key Files:
- **Predictions**: Detailed transcriptions for each model/condition
- **Comparisons**: Summary tables with CER and error analysis
- **Plots**: Visual comparison of model performance
- **Robustness Analysis**: Quantitative robustness metrics

## Project Structure

### Organized Scripts:
```
scripts/
├── utils/
│   ├── audio_utils.py      # Audio loading and processing
│   └── text_utils.py       # Text normalization and error analysis
├── models/
│   └── asr_models.py       # ASR model wrappers (Whisper, wav2vec2, Vosk)
├── audio_processing/
│   └── voice_segmentation.py  # Voice segment detection and modification
├── evaluation/
│   └── evaluate_models.py  # Model evaluation and results processing
└── __init__.py
```

### Key Features:
- **Modular Design**: Reusable components for different tasks
- **Error Analysis**: Detailed substitution/insertion/deletion breakdown
- **Visualization**: Automatic plot generation for results
- **Remote Ready**: Designed for cloud/remote server execution

## Models Supported

1. **Whisper** (`openai/whisper-small`)
   - State-of-the-art multilingual ASR
   - Good for general speech recognition

2. **wav2vec2** (`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`)
   - Specialized for Chinese
   - CTC-based architecture

3. **Vosk** (local model)
   - Lightweight offline ASR
   - Requires model download/upload

## Audio Processing

The pipeline automatically:
1. Detects voiced segments using pitch detection
2. Identifies elongated segments (>0.5s)
3. Applies time-stretching to shorten by 25%
4. Preserves unvoiced segments and short utterances

## Evaluation Metrics

- **CER**: Character Error Rate
- **Error Types**: Substitution, Insertion, Deletion rates
- **Robustness Score**: Sensitivity to audio modifications
- **Improvement**: Performance change between conditions

## Troubleshooting

### Common Issues:

1. **Missing Audio Files**
   - Check file paths in CSV
   - Verify upload completed successfully

2. **CUDA Out of Memory**
   - Set `DEVICE = "cpu"` in configuration
   - Reduce batch size or use smaller model

3. **Vosk Model Issues**
   - Remove "vosk" from `models_to_eval` list
   - Verify model path is correct

4. **Import Errors**
   - Run the environment setup cell
   - Install missing packages

### Performance Tips:

- Use GPU for faster evaluation (if available)
- Process smaller datasets first for testing
- Disable verbose output for cleaner logs

## Support

For issues or questions:
1. Check the notebook cell outputs for error messages
2. Verify file paths and permissions
3. Ensure all dependencies are installed
4. Test with a small dataset first

---

**🎯 Ready to evaluate ASR robustness on Chinese opera singing!**
