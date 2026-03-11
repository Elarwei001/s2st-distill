"""
Language-pair and layer pruning utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from contextlib import contextmanager
from tqdm import tqdm


class LanguagePairPruner:
    """
    Prune model to support only a single language pair.
    
    Removes unused language embeddings and freezes irrelevant parameters.
    """
    
    def __init__(self, source_lang: str = "eng", target_lang: str = "cmn"):
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def prune(self, model: nn.Module) -> nn.Module:
        """
        Prune model for single language pair.
        
        Args:
            model: Original multilingual model
        
        Returns:
            Pruned model
        """
        # Get language IDs
        src_id, tgt_id = self._get_language_ids(model)
        
        # Freeze unused language embeddings
        model = self._freeze_unused_embeddings(model, src_id, tgt_id)
        
        # Optionally remove unused vocabulary
        # model = self._prune_vocabulary(model, tgt_id)
        
        return model
    
    def _get_language_ids(self, model) -> tuple:
        """Get language IDs from model config."""
        config = model.config
        
        # Try different attribute names
        lang_to_id = getattr(config, "lang_code_to_id", None)
        if lang_to_id is None:
            lang_to_id = getattr(config, "language_to_id", {})
        
        src_id = lang_to_id.get(self.source_lang, 0)
        tgt_id = lang_to_id.get(self.target_lang, 1)
        
        return src_id, tgt_id
    
    def _freeze_unused_embeddings(
        self,
        model: nn.Module,
        src_id: int,
        tgt_id: int
    ) -> nn.Module:
        """Freeze embedding parameters for unused languages."""
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lang_embed" in name:
                # Create mask for used languages
                # For now, just mark as requiring gradient
                # Full pruning requires vocabulary analysis
                pass
        
        return model
    
    def _prune_vocabulary(
        self,
        model: nn.Module,
        tgt_lang_id: int
    ) -> nn.Module:
        """
        Prune vocabulary to only target language tokens.
        
        Warning: This is a destructive operation that changes model architecture.
        """
        # This requires detailed vocabulary analysis
        # Implementation depends on tokenizer structure
        
        # Placeholder - return model unchanged
        return model


class LayerPruner:
    """
    Iteratively prune layers based on importance scores.
    
    Reference: CULL-MT (https://arxiv.org/abs/2411.06506)
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
    
    def prune(
        self,
        model: nn.Module,
        dataloader,
        target_layers: int = 8,
        fine_tune_epochs: int = 2,
        num_eval_samples: int = 100
    ) -> nn.Module:
        """
        Iteratively prune layers to target count.
        
        Args:
            model: Model to prune
            dataloader: DataLoader for importance computation
            target_layers: Target number of layers
            fine_tune_epochs: Epochs to fine-tune after each pruning
            num_eval_samples: Samples for importance evaluation
        
        Returns:
            Pruned model
        """
        model = model.to(self.device)
        
        # Get current layer count
        encoder_layers = self._get_layer_list(model, "encoder")
        decoder_layers = self._get_layer_list(model, "decoder")
        
        current_encoder_layers = len(encoder_layers)
        current_decoder_layers = len(decoder_layers)
        
        print(f"Current layers: encoder={current_encoder_layers}, decoder={current_decoder_layers}")
        print(f"Target layers: {target_layers}")
        
        # Prune encoder
        while current_encoder_layers > target_layers:
            print(f"\nPruning encoder: {current_encoder_layers} → {current_encoder_layers - 1}")
            
            importance = self._compute_layer_importance(
                model, dataloader, "encoder", num_eval_samples
            )
            
            least_important = min(importance, key=importance.get)
            print(f"Removing encoder layer {least_important} (importance: {importance[least_important]:.4f})")
            
            model = self._remove_layer(model, "encoder", least_important)
            model = self._fine_tune(model, dataloader, fine_tune_epochs)
            
            current_encoder_layers -= 1
        
        # Prune decoder
        while current_decoder_layers > target_layers:
            print(f"\nPruning decoder: {current_decoder_layers} → {current_decoder_layers - 1}")
            
            importance = self._compute_layer_importance(
                model, dataloader, "decoder", num_eval_samples
            )
            
            least_important = min(importance, key=importance.get)
            print(f"Removing decoder layer {least_important} (importance: {importance[least_important]:.4f})")
            
            model = self._remove_layer(model, "decoder", least_important)
            model = self._fine_tune(model, dataloader, fine_tune_epochs)
            
            current_decoder_layers -= 1
        
        return model
    
    def _get_layer_list(self, model: nn.Module, component: str) -> nn.ModuleList:
        """Get layer list from model component."""
        if component == "encoder":
            if hasattr(model, "encoder"):
                return model.encoder.layers
            elif hasattr(model, "speech_encoder"):
                return model.speech_encoder.layers
        else:  # decoder
            if hasattr(model, "decoder"):
                return model.decoder.layers
            elif hasattr(model, "text_decoder"):
                return model.text_decoder.layers
        
        raise ValueError(f"Cannot find {component} layers in model")
    
    def _compute_layer_importance(
        self,
        model: nn.Module,
        dataloader,
        component: str,
        num_samples: int
    ) -> Dict[int, float]:
        """
        Compute importance score for each layer.
        
        Higher score = more important (bigger loss increase when removed).
        """
        model.eval()
        layers = self._get_layer_list(model, component)
        num_layers = len(layers)
        
        # Compute baseline loss
        baseline_loss = self._evaluate_loss(model, dataloader, num_samples)
        
        importance = {}
        for layer_idx in range(num_layers):
            with self._temporarily_disable_layer(layers, layer_idx):
                layer_loss = self._evaluate_loss(model, dataloader, num_samples)
            
            importance[layer_idx] = layer_loss - baseline_loss
        
        return importance
    
    @contextmanager
    def _temporarily_disable_layer(self, layers: nn.ModuleList, layer_idx: int):
        """Temporarily disable a layer by replacing with identity."""
        layer = layers[layer_idx]
        original_forward = layer.forward
        
        # Replace with identity function
        def identity_forward(x, *args, **kwargs):
            return x
        
        layer.forward = identity_forward
        
        try:
            yield
        finally:
            layer.forward = original_forward
    
    def _evaluate_loss(
        self,
        model: nn.Module,
        dataloader,
        num_samples: int
    ) -> float:
        """Compute average loss on samples."""
        model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if count >= num_samples:
                    break
                
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = model(audio)
                loss = F.cross_entropy(outputs.logits, labels)
                total_loss += loss.item()
                count += 1
        
        return total_loss / max(count, 1)
    
    def _remove_layer(
        self,
        model: nn.Module,
        component: str,
        layer_idx: int
    ) -> nn.Module:
        """Remove a layer from the model."""
        layers = self._get_layer_list(model, component)
        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(layers) if i != layer_idx
        ])
        
        # Update model
        if component == "encoder":
            if hasattr(model, "encoder"):
                model.encoder.layers = new_layers
            elif hasattr(model, "speech_encoder"):
                model.speech_encoder.layers = new_layers
        else:
            if hasattr(model, "decoder"):
                model.decoder.layers = new_layers
            elif hasattr(model, "text_decoder"):
                model.text_decoder.layers = new_layers
        
        return model
    
    def _fine_tune(
        self,
        model: nn.Module,
        dataloader,
        epochs: int = 2
    ) -> nn.Module:
        """Quick fine-tuning after layer removal."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        model.train()
        
        for epoch in range(epochs):
            for batch in dataloader:
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = model(audio)
                loss = F.cross_entropy(outputs.logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return model
