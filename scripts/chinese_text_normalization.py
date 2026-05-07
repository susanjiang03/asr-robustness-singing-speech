"""
Chinese text normalization and Whisper Mandarin transcription utilities
"""
import re
import jieba
from opencc import OpenCC

class ChineseTextNormalizer:
    """Normalize Chinese text and force Whisper Mandarin transcription"""
    
    def __init__(self):
        # Initialize OpenCC converter (Traditional to Simplified)
        self.cc = OpenCC('t2s')
        
        # Initialize jieba for Chinese word segmentation
        jieba.initialize()
        
        # Common Chinese punctuation mappings
        self.punctuation_map = {
            '，': ',',  # Full-width comma
            '。': '.',  # Full-width period
            '？': '?',  # Full-width question mark
            '！': '!',  # Full-width exclamation mark
            '：': ':',  # Full-width colon
            '；': ';',  # Full-width semicolon
            '（': '(',  # Full-width left parenthesis
            '）': ')',  # Full-width right parenthesis
            '【': '[',  # Full-width left bracket
            '】': ']',  # Full-width right bracket
            '"': '"',  # Full-width quotation mark
            '"': '"',  # Full-width quotation mark
        }
        
        # Common Chinese character normalization
        self.char_map = {
            '〇': '零',
            '一': '1',
            '二': '2', 
            '三': '3',
            '四': '4',
            '五': '5',
            '六': '6',
            '七': '7',
            '八': '8',
            '九': '9',
            '十': '10',
        }
    
    def normalize_chinese_text(self, text: str) -> str:
        """Normalize Chinese text to simplified characters and standard punctuation"""
        if not text:
            return ""
        
        # Convert Traditional to Simplified Chinese
        text = self.cc.convert(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', '', text.strip())
        
        # Normalize full-width punctuation to half-width
        for full_width, half_width in self.punctuation_map.items():
            text = text.replace(full_width, half_width)
        
        # Normalize Chinese numerals to Arabic numerals
        for chinese_num, arabic_num in self.char_map.items():
            text = text.replace(chinese_num, arabic_num)
        
        return text
    
    def extract_chinese_chars(self, text: str) -> str:
        """Extract only Chinese characters (including punctuation)"""
        # Keep Chinese characters, punctuation, and common symbols
        chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U000f900-\U000faff\U0003300-\U00033ff\ufe30-\ufe4f\uf900-\ufaff\uff00-\uffef，。？！：；（）【】""''""''…—～·]'
        return ''.join(re.findall(chinese_pattern, text))
    
    def post_process_whisper_output(self, text: str) -> str:
        """Post-process Whisper output for Chinese text"""
        if not text:
            return ""
        
        # Normalize the text
        text = self.normalize_chinese_text(text)
        
        # Remove Whisper artifacts and common errors
        # Remove repeated characters (common in Whisper hallucinations)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Remove English letters that might appear in Chinese text
        text = re.sub(r'[a-zA-Z]+', '', text)
        
        # Fix common Whisper Chinese errors
        common_fixes = {
            '的得': '的',
            '的了': '了',
            '和和': '和',
            '在在': '在',
            '是有': '是',
            '不不': '不',
            '这这': '这',
            '那那': '那',
            '哪那': '哪',
            '吗嘛': '吗',
        }
        
        for wrong, correct in common_fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def validate_chinese_text(self, text: str) -> dict:
        """Validate Chinese text quality"""
        if not text:
            return {"valid": False, "reason": "Empty text"}
        
        # Count Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)
        
        # Calculate Chinese character ratio
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        
        # Check for common issues
        issues = []
        if chinese_ratio < 0.5:
            issues.append(f"Low Chinese character ratio: {chinese_ratio:.2f}")
        
        # Check for repeated characters (hallucination indicator)
        if re.search(r'(.)\1{3,}', text):
            issues.append("Repeated characters detected")
        
        # Check for English letters
        if re.search(r'[a-zA-Z]', text):
            issues.append("English letters present")
        
        return {
            "valid": len(issues) == 0,
            "chinese_ratio": chinese_ratio,
            "total_chars": total_chars,
            "chinese_chars": chinese_chars,
            "issues": issues
        }
    
    def force_mandarin_transcription_config(self) -> dict:
        """Get Whisper configuration for Mandarin transcription"""
        return {
            "language": "zh",
            "task": "transcribe",
            "initial_prompt": "以下是中文普通话的句子：",  # Force Chinese context
            "condition_on_previous_text": True,
            "temperature": 0.0,  # Lower temperature for more deterministic output
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "suppress_tokens": [],  # Don't suppress Chinese tokens
            "prepend_punctuations": "，。？！：；",
            "append_punctuations": "，。？！：；",
        }


def normalize_prediction_for_evaluation(prediction: str, reference: str) -> tuple[str, str]:
    """
    Normalize both prediction and reference for fair evaluation
    
    Args:
        prediction: Whisper output text
        reference: Ground truth text
    
    Returns:
        tuple: (normalized_prediction, normalized_reference)
    """
    normalizer = ChineseTextNormalizer()
    
    # Normalize both texts
    norm_pred = normalizer.normalize_chinese_text(prediction)
    norm_ref = normalizer.normalize_chinese_text(reference)
    
    # Post-process Whisper output
    norm_pred = normalizer.post_process_whisper_output(norm_pred)
    
    return norm_pred, norm_ref


def evaluate_chinese_asr(predictions: list, references: list) -> dict:
    """
    Evaluate Chinese ASR with proper normalization
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        dict: Evaluation metrics
    """
    normalizer = ChineseTextNormalizer()
    
    normalized_predictions = []
    normalized_references = []
    
    # Normalize all predictions and references
    for pred, ref in zip(predictions, references):
        norm_pred = normalizer.post_process_whisper_output(pred)
        norm_ref = normalizer.normalize_chinese_text(ref)
        
        normalized_predictions.append(norm_pred)
        normalized_references.append(norm_ref)
    
    # Calculate character-level metrics for Chinese
    char_errors = 0
    char_total = 0
    
    for pred, ref in zip(normalized_predictions, normalized_references):
        # Character Error Rate for Chinese
        errors = calculate_cer_chinese(pred, ref)
        char_errors += errors['errors']
        char_total += errors['total']
    
    cer = char_errors / char_total if char_total > 0 else 0
    
    return {
        "cer": cer,
        "char_errors": char_errors,
        "char_total": char_total,
        "normalized_predictions": normalized_predictions,
        "normalized_references": normalized_references,
    }


def calculate_cer_chinese(prediction: str, reference: str) -> dict:
    """
    Calculate Character Error Rate for Chinese text
    
    Args:
        prediction: Predicted Chinese text
        reference: Reference Chinese text
    
    Returns:
        dict: CER calculation details
    """
    if not reference:
        return {"errors": 0, "total": 0, "cer": 0}
    
    # Normalize both texts
    normalizer = ChineseTextNormalizer()
    pred = normalizer.normalize_chinese_text(prediction)
    ref = normalizer.normalize_chinese_text(reference)
    
    # Calculate character-level errors
    # Use dynamic programming for optimal alignment
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == ref[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost   # substitution
            )
    
    errors = dp[m][n]
    total = len(ref)
    
    return {
        "errors": errors,
        "total": total,
        "cer": errors / total if total > 0 else 0
    }


# Whisper Mandarin transcription configuration
MANDARIN_CONFIG = ChineseTextNormalizer().force_mandarin_transcription_config()

if __name__ == "__main__":
    # Test the normalizer
    normalizer = ChineseTextNormalizer()
    
    test_texts = [
        "这是一个测试文本，包含全角标点符号！",
        "Traditional Chinese: 繁體中文",
        "Mixed 123 Numbers and English text",
        "重复重复重复的字符",
    ]
    
    print("Chinese Text Normalization Test:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {text}")
        
        # Normalize text
        normalized = normalizer.normalize_chinese_text(text)
        print(f"Normalized: {normalized}")
        
        # Post-process Whisper output
        post_processed = normalizer.post_process_whisper_output(text)
        print(f"Post-processed: {post_processed}")
        
        # Validate text
        validation = normalizer.validate_chinese_text(text)
        print(f"Validation: {validation}")
    
    print(f"\nMandarin Config: {MANDARIN_CONFIG}")
