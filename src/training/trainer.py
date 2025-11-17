# src/training/trainer.py
"""
Training Pipeline for CVRP GNN
===============================
Sistema completo di training con validazione e logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time
import json
import torch.nn.functional as F

class CVRPLoss(nn.Module):
    """Loss combinata per CVRP"""
    
    def __init__(self,
                 edge_weight: float = 1.0,
                 node_weight: float = 0.5,
                 consistency_weight: float = 0.2):
        super().__init__()
        self.edge_weight = edge_weight
        self.node_weight = node_weight
        self.consistency_weight = consistency_weight
        
        self.register_buffer('ema_ratio', torch.tensor(1.0))
        self.ema_beta = 0.9  # più alto = più stabile
        self.pos_weight_min = 5.0
        self.pos_weight_max = 5000.0

        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: Dict, data) -> Dict[str, torch.Tensor]:
        """
        Calcola le loss combinate.
        
        Returns:
            Dict con loss totale e componenti
        """
        losses = {}
        
        # Edge loss (classificazione binaria)
        if hasattr(data, 'y_edges'):
            y_true = data.y_edges
            # count pos/neg nel batch
            N_pos = (y_true == 1).sum()
            N_neg = (y_true == 0).sum()
        
            # evita div/zero
            ratio = (N_neg.float() / torch.clamp(N_pos.float(), min=1.0))
        
            # EMA + clamp per stabilità
            self.ema_ratio = self.ema_beta * self.ema_ratio + (1 - self.ema_beta) * ratio.detach()
            ratio_clamped = torch.clamp(self.ema_ratio, min=self.pos_weight_min, max=self.pos_weight_max)
        
            pos_weight = ratio_clamped.to(y_true.device).unsqueeze(0)  # shape [1]
        
            bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            edge_loss = bce_loss_fn(predictions['edge_predictions'], y_true)
        
            losses['edge_loss'] = edge_loss * self.edge_weight

        
        # Node sequence loss
        if hasattr(data, 'y_nodes'):
            mask = data.y_nodes >= 0  # Ignora nodi non visitati
            if mask.any():
                node_loss = self.mse_loss(
                    predictions['node_predictions'][mask],
                    data.y_nodes[mask]
                )
                losses['node_loss'] = node_loss * self.node_weight
        
        # Consistency loss (edges e nodes devono essere coerenti)
        if 'edge_predictions' in predictions and 'node_predictions' in predictions:
            consistency_loss = self._consistency_loss(
                predictions['edge_predictions'],
                predictions['node_predictions'],
                data.edge_index
            )
            losses['consistency_loss'] = consistency_loss * self.consistency_weight
        
        # Loss totale
        losses['total_loss'] = sum(losses.values())
        
        return losses
    
    def _consistency_loss(self, edge_preds, node_preds, edge_index):
        """Penalizza incoerenze tra predizioni edges e nodes"""
        edge_probs = torch.sigmoid(edge_preds)
        
        # Nodi con sequence alta dovrebbero avere edges con prob alta
        src_seq = node_preds[edge_index[0]]
        dst_seq = node_preds[edge_index[1]]
        
        # Edges tra nodi consecutivi dovrebbero avere prob alta
        seq_diff = torch.abs(dst_seq - src_seq - 1)
        expected_prob = torch.exp(-seq_diff)
        
        return F.mse_loss(edge_probs, expected_prob)


class CVRPTrainer:
    """Trainer per modelli CVRP GNN"""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: Optional[Path] = None):
        """
        Args:
            model: Modello GNN
            optimizer: Ottimizzatore
            loss_fn: Funzione di loss
            device: Device per training
            checkpoint_dir: Directory per salvare checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        print(f"🚀 Trainer inizializzato su {device}")
    
    def train_epoch(self, train_loader: GeometricDataLoader) -> Dict[str, float]:
        """Esegue un'epoca di training"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'edge_acc': [],
            'node_mae': [],
            'pos_rate@0.5': [],
            'mean_logit': [],
            'mean_prob': []
        }
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Calcola loss
            losses = self.loss_fn(predictions, batch)
            total_loss = losses['total_loss']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Tracking
            epoch_losses.append(total_loss.item())
            
            # Calcola metriche
            with torch.no_grad():
                metrics = self._compute_metrics(predictions, batch)
                for key, value in metrics.items():
                    if value is not None:
                        epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'edge_acc': f"{metrics.get('edge_acc', 0):.2%}"
            })
        
        # Calcola medie
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in epoch_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate(self, val_loader: GeometricDataLoader) -> Dict[str, float]:
        """Valida il modello"""
        self.model.eval()
        val_losses = []
        val_metrics = {
            'edge_acc': [],
            'node_mae': [],
            'route_quality': [],
            'pos_rate@0.5': [],
            'mean_logit': [],
            'mean_prob': []
        }

        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Loss
                losses = self.loss_fn(predictions, batch)
                val_losses.append(losses['total_loss'].item())
                
                # Metriche
                metrics = self._compute_metrics(predictions, batch)
                for key, value in metrics.items():
                    if value is not None:
                        val_metrics[key].append(value)
        
        # Calcola medie
        avg_loss = np.mean(val_losses)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in val_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def _compute_metrics(self, predictions: Dict, batch) -> Dict[str, float]:
        """Calcola metriche di valutazione"""
        metrics = {}
        
        # Edge accuracy Claude
        # if 'edge_predictions' in predictions and hasattr(batch, 'y_edges'):
        #     edge_probs = torch.sigmoid(predictions['edge_predictions'])
        #     edge_preds = (edge_probs > 0.5).float()
        #     correct = (edge_preds == batch.y_edges).float()
        #     metrics['edge_acc'] = correct.mean().item()
        
        # Node MAE
        if 'node_predictions' in predictions and hasattr(batch, 'y_nodes'):
            mask = batch.y_nodes >= 0
            if mask.any():
                mae = torch.abs(
                    predictions['node_predictions'][mask] - batch.y_nodes[mask]
                ).mean()
                metrics['node_mae'] = mae.item()
                
        if 'edge_predictions' in predictions and hasattr(batch, 'y_edges'):
            edge_logits = predictions['edge_predictions']
            edge_probs  = torch.sigmoid(edge_logits)
            edge_preds  = (edge_probs > 0.5).float()
        
            correct = (edge_preds == batch.y_edges).float()
            metrics['edge_acc'] = correct.mean().item()
        
            # logging extra
            metrics['pos_rate@0.5'] = (edge_preds.mean()).item()  # % archi predetti positivi
            with torch.no_grad():
                metrics['mean_logit'] = edge_logits.mean().item()
                metrics['mean_prob']  = edge_probs.mean().item()

        
        return metrics
    
    def train(self,
             train_loader: GeometricDataLoader,
             val_loader: GeometricDataLoader,
             epochs: int,
             patience: int = 10,
             scheduler: Optional[object] = None) -> Dict:
        """
        Training loop completo.
        
        Args:
            train_loader: DataLoader per training
            val_loader: DataLoader per validation
            epochs: Numero di epoche
            patience: Early stopping patience
            scheduler: Learning rate scheduler opzionale
            
        Returns:
            Storia del training
        """
        print("\n" + "="*60)
        print("🎯 INIZIO TRAINING")
        print("="*60)
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Training
            train_stats = self.train_epoch(train_loader)
            
            # Validation
            val_stats = self.validate(val_loader)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step(val_stats['loss'])
            
            # Logging
            epoch_time = time.time() - start_time
            self._log_epoch(epoch, train_stats, val_stats, epoch_time)
            
            # Salva history
            self.history['train_loss'].append(train_stats['loss'])
            self.history['val_loss'].append(val_stats['loss'])
            self.history['train_metrics'].append(train_stats)
            self.history['val_metrics'].append(val_stats)
            
            # Checkpoint se miglioramento
            if val_stats['loss'] < self.best_val_loss:
                self.best_val_loss = val_stats['loss']
                self.patience_counter = 0
                
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch, val_stats['loss'])
                    print(f"   💾 Checkpoint salvato (best loss: {val_stats['loss']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n⚠️ Early stopping dopo {epoch} epoche")
                    break
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETATO")
        print("="*60)
        print(f"📊 Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def _log_epoch(self, epoch: int, train_stats: Dict, val_stats: Dict, epoch_time: float):
        """Log delle statistiche per epoca"""
        print(f"\n📈 Epoca {epoch}")
        print(f"   ⏱️  Tempo: {epoch_time:.1f}s")
        print(f"   📉 Train Loss: {train_stats['loss']:.4f} | Val Loss: {val_stats['loss']:.4f}")
        print(f"   🎯 Train Edge Acc: {train_stats.get('edge_acc', 0):.2%} | "
              f"Val Edge Acc: {val_stats.get('edge_acc', 0):.2%}")
        
        if 'node_mae' in train_stats:
            print(f"   📏 Train Node MAE: {train_stats['node_mae']:.2f} | "
                  f"Val Node MAE: {val_stats.get('node_mae', 0):.2f}")
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Salva checkpoint del modello"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'config': {
                'model_class': self.model.__class__.__name__,
                'device': str(self.device)
            }
        }
        
        path = self.checkpoint_dir / f"best_model.pth"
        torch.save(checkpoint, path)
        
        # Salva anche history come JSON
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)