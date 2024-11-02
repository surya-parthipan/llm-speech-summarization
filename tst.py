import torch
import torch.nn as nn
import torchaudio
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoModelForSeq2SeqLM, AutoTokenizer

# Define the EmotionRecognitionModel within the same script
class EmotionRecognitionModel(nn.Module):
    def __init__(self, pretrained_model_name="facebook/wav2vec2-large-960h"):
        super(EmotionRecognitionModel, self).__init__()
        self.feature_extractor = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # Assuming 8 basic emotion classes (happy, sad, angry, etc.)
        )

    def forward(self, audio_input):
        features = self.feature_extractor(audio_input).last_hidden_state
        pooled_features = torch.mean(features, dim=1)
        emotion_logits = self.classifier(pooled_features)
        return emotion_logits

    def extract_emotion_embedding(self, audio_input):
        with torch.no_grad():
            logits = self.forward(audio_input)
            probabilities = torch.softmax(logits, dim=-1)
            return probabilities

# Define the AudioFeatureExtractor within the same script
class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=40):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate, n_mfcc=self.n_mfcc, melkwargs={"n_mels": 64, "hop_length": 512}
        )

    def extract_features(self, audio_signal):
        mfcc = self.mfcc_transform(audio_signal)
        pooled_mfcc = torch.mean(mfcc, dim=-1)
        return pooled_mfcc

    def __call__(self, audio_signal):
        return self.extract_features(audio_signal)

# Main pipeline class
class AudioSummarizationPipeline:
    def __init__(self, audio_model_checkpoint, emotion_model_checkpoint, text_model_name, device):
        self.device = device

        # Initialize the audio feature extractor
        self.audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_checkpoint)
        self.audio_model = Wav2Vec2ForCTC.from_pretrained(audio_model_checkpoint).to(self.device)

        # Initialize the emotion recognition model
        self.emotion_recognizer = EmotionRecognitionModel(emotion_model_checkpoint).to(self.device)

        # Initialize the text summarization model (LLM)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModelForSeq2SeqLM.from_pretrained(text_model_name).to(self.device)

        # Initialize the custom audio feature extractor
        self.audio_feature_extractor = AudioFeatureExtractor()

    def extract_text_from_audio(self, audio):
        inputs = self.audio_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.audio_model(inputs.input_values.to(self.device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.audio_processor.batch_decode(predicted_ids)[0].lower()
        return transcription

    def extract_emotion_from_audio(self, audio):
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        emotion_embedding = self.emotion_recognizer.extract_emotion_embedding(audio_tensor)
        return emotion_embedding

    def encode_audio_features(self, audio):
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        audio_features = self.audio_feature_extractor(audio_tensor)
        return audio_features

    def generate_summary(self, audio, additional_context="", max_new_tokens=256):
        transcription = self.extract_text_from_audio(audio)
        emotion_embedding = self.extract_emotion_from_audio(audio)
        audio_features = self.encode_audio_features(audio)

        # Combining all context, transcription, and extracted features
        full_context = f"{additional_context} {transcription}".strip()
        tokenized_input = self.text_tokenizer(full_context, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            summary_ids = self.text_model.generate(
                input_ids=tokenized_input,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True
            )

        summary = self.text_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = AudioSummarizationPipeline(
        audio_model_checkpoint="facebook/wav2vec2-large-960h",
        emotion_model_checkpoint="facebook/wav2vec2-large-960h",  # Same model is used in this example
        text_model_name="t5-base",
        device=device
    )

    # Load and preprocess the audio file
    audio, sr = librosa.load("article2.wav", sr=16000)
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)  # Reshaping for model compatibility

    # Generate summary from the audio
    summary = pipeline.generate_summary(audio, max_new_tokens=512)
    print(f"Generated Summary: {summary}")
