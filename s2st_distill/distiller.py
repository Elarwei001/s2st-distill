"""
Main distillation pipeline for S2ST models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, SeamlessM4TModel
from tqdm import tqdm
from typing import Optional, Dict, Any

from .pruning import LanguagePairPruner, LayerPruner
from .quantize import quantize_int8
from .export import export_onnx, export_coreml, export_tflite


class DistillationLoss(nn.Module):
    """Combined knowledge distillation and hard label loss."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        # Soft target loss (knowledge distillation)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Hard target loss (ground truth labels)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class S2STDistiller:
    """
    Main class for distilling S2ST models.
    
    Example:
        distiller = S2STDistiller(
            base_model="facebook/seamless-m4t-unity-small",
            source_lang="eng",
            target_lang="cmn"
        )
        student = distiller.distill(train_dataset, num_epochs=10)
        distiller.export_coreml("model.mlpackage")
    """
    
    def __init__(
        self,
        base_model: str = "facebook/seamless-m4t-unity-small",
        source_lang: str = "eng",
        target_lang: str = "cmn",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.base_model_name = base_model
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device
        
        # Load models
        print(f"Loading base model: {base_model}")
        self.processor = AutoProcessor.from_pretrained(base_model)
        self.teacher = SeamlessM4TModel.from_pretrained(base_model).to(device)
        self.student = SeamlessM4TModel.from_pretrained(base_model).to(device)
        
        # Initialize pruners
        self.lang_pruner = LanguagePairPruner(source_lang, target_lang)
        self.layer_pruner = LayerPruner()
        
        print(f"Initialized S2STDistiller for {source_lang} → {target_lang}")
    
    def distill(
        self,
        train_dataset,
        val_dataset=None,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        temperature: float = 4.0,
        alpha: float = 0.7,
        target_layers: int = 8,
        target_size_mb: Optional[float] = None,
    ) -> nn.Module:
        """
        Run the full distillation pipeline.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            temperature: Distillation temperature
            alpha: Weight for soft loss (1-alpha for hard loss)
            target_layers: Target number of encoder/decoder layers
            target_size_mb: Target model size in MB
        
        Returns:
            Distilled student model
        """
        # Phase 1: Language-pair pruning
        print("\n=== Phase 1: Language-Pair Pruning ===")
        self.student = self.lang_pruner.prune(self.student)
        self._print_model_size("After language pruning")
        
        # Phase 2: Knowledge distillation
        print("\n=== Phase 2: Knowledge Distillation ===")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        self.student = self._train_distillation(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            temperature=temperature,
            alpha=alpha
        )
        
        # Phase 3: Layer pruning
        print("\n=== Phase 3: Layer Pruning ===")
        self.student = self.layer_pruner.prune(
            self.student,
            train_loader,
            target_layers=target_layers
        )
        self._print_model_size("After layer pruning")
        
        # Phase 4: Fine-tune after pruning
        print("\n=== Phase 4: Post-Pruning Fine-tuning ===")
        self.student = self._fine_tune(train_loader, epochs=2)
        
        # Phase 5: Quantization
        print("\n=== Phase 5: Quantization ===")
        self.student = quantize_int8(self.student)
        self._print_model_size("After quantization")
        
        return self.student
    
    def _train_distillation(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int,
        learning_rate: float,
        temperature: float,
        alpha: float
    ) -> nn.Module:
        """Train student with knowledge distillation."""
        self.teacher.eval()
        self.student.train()
        
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        criterion = DistillationLoss(temperature, alpha)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.student.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Teacher forward (no gradient)
                with torch.no_grad():
                    teacher_outputs = self.teacher(audio)
                    teacher_logits = teacher_outputs.logits
                
                # Student forward
                student_outputs = self.student(audio)
                student_logits = student_outputs.logits
                
                # Compute loss
                loss = criterion(student_logits, teacher_logits, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader:
                val_loss = self._evaluate(val_loader)
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.student.state_dict(), "best_student.pt")
            else:
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")
            
            scheduler.step()
        
        return self.student
    
    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on dataloader."""
        self.student.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.student(audio)
                loss = F.cross_entropy(outputs.logits, labels)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _fine_tune(self, dataloader: DataLoader, epochs: int = 2) -> nn.Module:
        """Quick fine-tuning after pruning."""
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=1e-5)
        self.student.train()
        
        for epoch in range(epochs):
            for batch in tqdm(dataloader, desc=f"Fine-tune {epoch+1}/{epochs}"):
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.student(audio)
                loss = F.cross_entropy(outputs.logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self.student
    
    def _print_model_size(self, stage: str):
        """Print current model size."""
        total_params = sum(p.numel() for p in self.student.parameters())
        trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        
        # Estimate size (assuming float32)
        size_mb = total_params * 4 / 1e6
        
        print(f"{stage}:")
        print(f"  Total params: {total_params / 1e6:.1f}M")
        print(f"  Trainable params: {trainable_params / 1e6:.1f}M")
        print(f"  Estimated size: {size_mb:.1f} MB")
    
    def export_onnx(self, output_path: str = "model.onnx"):
        """Export model to ONNX format."""
        export_onnx(self.student, output_path)
    
    def export_coreml(self, output_path: str = "model.mlpackage"):
        """Export model to CoreML format for iOS."""
        onnx_path = output_path.replace(".mlpackage", ".onnx")
        export_onnx(self.student, onnx_path)
        export_coreml(onnx_path, output_path)
    
    def export_tflite(self, output_path: str = "model.tflite"):
        """Export model to TFLite format for Android."""
        onnx_path = output_path.replace(".tflite", ".onnx")
        export_onnx(self.student, onnx_path)
        export_tflite(onnx_path, output_path)
