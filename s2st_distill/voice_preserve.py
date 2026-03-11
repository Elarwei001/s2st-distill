"""
Voice preservation modules: Speaker encoder and prosody transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    """
    Extract speaker embeddings to preserve voice identity.
    
    Architecture inspired by ECAPA-TDNN for robust speaker representation.
    """
    
    def __init__(self, input_dim: int = 80, embed_dim: int = 256):
        """
        Args:
            input_dim: Mel spectrogram dimension (default 80)
            embed_dim: Output speaker embedding dimension
        """
        super().__init__()
        
        # Frame-level feature extraction (1D CNN)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        # Attentive statistics pooling
        self.attention = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 512, kernel_size=1),
            nn.Softmax(dim=2),
        )
        
        # Final embedding projection
        self.fc = nn.Linear(512 * 2, embed_dim)  # Mean + Std
    
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from mel spectrogram.
        
        Args:
            mel_spectrogram: [B, T, 80] mel features
        
        Returns:
            speaker_embedding: [B, embed_dim] normalized embedding
        """
        # Transpose for Conv1d: [B, 80, T]
        x = mel_spectrogram.transpose(1, 2)
        
        # Encode frame-level features
        x = self.encoder(x)  # [B, 512, T]
        
        # Attentive pooling
        weights = self.attention(x)  # [B, 512, T]
        
        # Weighted statistics
        mean = (x * weights).sum(dim=2)  # [B, 512]
        std = torch.sqrt(
            (((x - mean.unsqueeze(2)) ** 2) * weights).sum(dim=2) + 1e-6
        )  # [B, 512]
        
        # Concatenate and project
        stats = torch.cat([mean, std], dim=1)  # [B, 1024]
        embedding = self.fc(stats)  # [B, embed_dim]
        
        # L2 normalize
        return F.normalize(embedding, dim=-1)


class ProsodyExtractor(nn.Module):
    """
    Extract prosodic features: pitch, duration, energy.
    
    These features capture rhythm, intonation, and emphasis.
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Pitch encoder (F0 contour)
        self.pitch_encoder = nn.LSTM(
            hidden_dim, 64, batch_first=True, bidirectional=True
        )
        
        # Duration encoder (speech rate, pauses)
        self.duration_encoder = nn.LSTM(
            hidden_dim, 64, batch_first=True, bidirectional=True
        )
        
        # Energy encoder (loudness, emphasis)
        self.energy_encoder = nn.LSTM(
            hidden_dim, 64, batch_first=True, bidirectional=True
        )
    
    def forward(self, hidden_states: torch.Tensor) -> dict:
        """
        Extract prosody features from hidden states.
        
        Args:
            hidden_states: [B, T, hidden_dim] encoder features
        
        Returns:
            dict with pitch, duration, energy tensors [B, T, 128]
        """
        pitch, _ = self.pitch_encoder(hidden_states)
        duration, _ = self.duration_encoder(hidden_states)
        energy, _ = self.energy_encoder(hidden_states)
        
        return {
            "pitch": pitch,      # [B, T, 128]
            "duration": duration,  # [B, T, 128]
            "energy": energy     # [B, T, 128]
        }


class ProsodyTransfer(nn.Module):
    """
    Transfer prosody from source speech to target translation.
    
    This module helps preserve the speaker's rhythm, intonation,
    and emotional expression in the translated speech.
    """
    
    def __init__(self, hidden_dim: int = 256, prosody_weight: float = 0.3):
        """
        Args:
            hidden_dim: Model hidden dimension
            prosody_weight: Weight for prosody contribution (0-1)
        """
        super().__init__()
        self.prosody_weight = prosody_weight
        
        # Prosody feature extractor
        self.prosody_extractor = ProsodyExtractor(hidden_dim)
        
        # Prosody adapter (384 = 128 * 3 features)
        self.prosody_adapter = nn.Sequential(
            nn.Linear(128 * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Cross-attention for length alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Transfer prosody from source to target.
        
        Args:
            source_features: [B, T_src, hidden_dim] source encoder output
            target_hidden: [B, T_tgt, hidden_dim] target decoder hidden
        
        Returns:
            enhanced_hidden: [B, T_tgt, hidden_dim] with prosody
        """
        # Extract source prosody
        prosody = self.prosody_extractor(source_features)
        
        # Combine prosody features
        prosody_combined = torch.cat([
            prosody["pitch"],
            prosody["duration"],
            prosody["energy"]
        ], dim=-1)  # [B, T_src, 384]
        
        # Adapt to hidden dimension
        prosody_adapted = self.prosody_adapter(prosody_combined)  # [B, T_src, hidden_dim]
        
        # Align to target length using cross-attention
        # Query: target, Key/Value: source prosody
        aligned_prosody, _ = self.cross_attention(
            query=target_hidden,
            key=prosody_adapted,
            value=prosody_adapted
        )  # [B, T_tgt, hidden_dim]
        
        # Add prosody to target (residual connection)
        enhanced = target_hidden + self.prosody_weight * aligned_prosody
        
        return enhanced


class VoicePreserver(nn.Module):
    """
    Combined module for preserving speaker identity and prosody.
    
    Integrates SpeakerEncoder and ProsodyTransfer for natural-sounding
    translated speech that maintains the original speaker's voice.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        speaker_embed_dim: int = 256,
        prosody_weight: float = 0.3
    ):
        super().__init__()
        
        self.speaker_encoder = SpeakerEncoder(embed_dim=speaker_embed_dim)
        self.prosody_transfer = ProsodyTransfer(hidden_dim, prosody_weight)
        
        # Speaker conditioning
        self.speaker_adapter = nn.Linear(speaker_embed_dim, hidden_dim)
    
    def forward(
        self,
        source_mel: torch.Tensor,
        source_features: torch.Tensor,
        target_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply voice preservation to target hidden states.
        
        Args:
            source_mel: [B, T, 80] source mel spectrogram
            source_features: [B, T_src, hidden_dim] source encoder output
            target_hidden: [B, T_tgt, hidden_dim] target decoder hidden
        
        Returns:
            preserved_hidden: [B, T_tgt, hidden_dim] with voice preservation
        """
        # Extract speaker embedding
        speaker_embed = self.speaker_encoder(source_mel)  # [B, speaker_embed_dim]
        speaker_cond = self.speaker_adapter(speaker_embed)  # [B, hidden_dim]
        
        # Add speaker conditioning (broadcast across time)
        target_with_speaker = target_hidden + speaker_cond.unsqueeze(1)
        
        # Transfer prosody
        output = self.prosody_transfer(source_features, target_with_speaker)
        
        return output
