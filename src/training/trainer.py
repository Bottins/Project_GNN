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
    """Loss combinata per CVRP con penalità per vincoli del dominio e Focal Loss"""

    def __init__(self,
                 edge_weight: float = 1.0,
                 node_weight: float = 0.3,
                 # Penalità MOLTO ridotte per non sopprimere le predizioni
                 self_loop_penalty: float = 0.5,
                 node_revisit_penalty: float = 0.3,
                 capacity_penalty: float = 0.8,
                 route_validity_penalty: float = 0.2,
                 # Focal loss parameters
                 use_focal_loss: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 # Warmup per penalità
                 penalty_warmup_epochs: int = 20):
        super().__init__()
        self.edge_weight = edge_weight
        self.node_weight = node_weight
        self.self_loop_penalty = self_loop_penalty
        self.node_revisit_penalty = node_revisit_penalty
        self.capacity_penalty = capacity_penalty
        self.route_validity_penalty = route_validity_penalty

        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.penalty_warmup_epochs = penalty_warmup_epochs

        # Traccia l'epoca corrente per warmup
        self.register_buffer('current_epoch', torch.tensor(0))

        self.register_buffer('ema_ratio', torch.tensor(1.0))
        self.ema_beta = 0.9
        self.pos_weight_min = 5.0
        self.pos_weight_max = 100.0  # Ridotto da 5000 a 100

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def set_epoch(self, epoch: int):
        """Imposta l'epoca corrente per il warmup delle penalità"""
        self.current_epoch = torch.tensor(epoch)

    def get_penalty_scale(self) -> float:
        """Calcola il fattore di scala per le penalità basato sull'epoca"""
        if self.current_epoch < self.penalty_warmup_epochs:
            # Warmup lineare da 0.1 a 1.0
            return 0.1 + 0.9 * (self.current_epoch.item() / self.penalty_warmup_epochs)
        return 1.0
    
    def forward(self, predictions: Dict, data) -> Dict[str, torch.Tensor]:
        """
        Calcola le loss combinate con penalità per vincoli VRP.

        Returns:
            Dict con loss totale e componenti
        """
        losses = {}
        penalty_scale = self.get_penalty_scale()

        # Edge loss (classificazione binaria con Focal Loss opzionale)
        if hasattr(data, 'y_edges'):
            y_true = data.y_edges

            if self.use_focal_loss:
                # Usa Focal Loss per gestire meglio lo sbilanciamento
                edge_loss = self._focal_loss(
                    predictions['edge_predictions'],
                    y_true,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma
                )
            else:
                # Fallback a BCE con pos_weight
                N_pos = (y_true == 1).sum()
                N_neg = (y_true == 0).sum()
                ratio = (N_neg.float() / torch.clamp(N_pos.float(), min=1.0))

                self.ema_ratio = self.ema_beta * self.ema_ratio + (1 - self.ema_beta) * ratio.detach()
                ratio_clamped = torch.clamp(self.ema_ratio, min=self.pos_weight_min, max=self.pos_weight_max)
                pos_weight = ratio_clamped.to(y_true.device).unsqueeze(0)

                bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                edge_loss = bce_loss_fn(predictions['edge_predictions'], y_true)

            losses['edge_loss'] = edge_loss * self.edge_weight

        # Node sequence loss (ridotto il peso)
        if hasattr(data, 'y_nodes'):
            mask = data.y_nodes >= 0
            if mask.any():
                node_loss = self.mse_loss(
                    predictions['node_predictions'][mask],
                    data.y_nodes[mask]
                )
                losses['node_loss'] = node_loss * self.node_weight

        # ===== PENALITÀ PER VINCOLI VRP (con warmup) =====

        # 1. Self-loop penalty
        if 'edge_predictions' in predictions:
            self_loop_loss = self._self_loop_penalty(
                predictions['edge_predictions'],
                data.edge_index
            )
            losses['self_loop_loss'] = self_loop_loss * self.self_loop_penalty * penalty_scale

        # 2. Node revisit penalty
        if 'edge_predictions' in predictions:
            node_revisit_loss = self._node_revisit_penalty(
                predictions['edge_predictions'],
                data.edge_index,
                data.num_nodes
            )
            losses['node_revisit_loss'] = node_revisit_loss * self.node_revisit_penalty * penalty_scale

        # 3. Capacity penalty (migliorata)
        if 'edge_predictions' in predictions and hasattr(data, 'x'):
            capacity_loss = self._capacity_penalty_v2(
                predictions['edge_predictions'],
                data.edge_index,
                data.x,
                data.capacity if hasattr(data, 'capacity') else None
            )
            losses['capacity_loss'] = capacity_loss * self.capacity_penalty * penalty_scale

        # 4. Route validity penalty
        if 'edge_predictions' in predictions:
            route_validity_loss = self._route_validity_penalty(
                predictions['edge_predictions'],
                data.edge_index,
                data.num_nodes
            )
            losses['route_validity_loss'] = route_validity_loss * self.route_validity_penalty * penalty_scale

        # Loss totale
        losses['total_loss'] = sum(losses.values())

        return losses

    def _focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """
        Focal Loss per gestire lo sbilanciamento delle classi.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Args:
            inputs: Logits predetti
            targets: Target binari (0 o 1)
            alpha: Peso per classe positiva
            gamma: Focusing parameter (gamma > 0 riduce peso agli esempi facili)
        """
        # Calcola BCE
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calcola probabilità
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calcola alpha_t
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

        # Focal term
        focal_weight = (1 - p_t) ** gamma

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean()
    
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

    def _self_loop_penalty(self, edge_preds, edge_index):
        """
        Penalizza fortemente archi self-loop (stesso nodo iniziale e finale).
        In un VRP, un veicolo non può andare da un nodo a se stesso.
        """
        # Identifica self-loops
        self_loop_mask = edge_index[0] == edge_index[1]

        if self_loop_mask.any():
            # Penalizza probabilità alta su self-loops
            self_loop_probs = torch.sigmoid(edge_preds[self_loop_mask])
            # Vogliamo che queste probabilità siano vicine a 0
            return torch.mean(self_loop_probs ** 2)
        else:
            return torch.tensor(0.0, device=edge_preds.device)

    def _node_revisit_penalty(self, edge_preds, edge_index, num_nodes):
        """
        Penalizza quando lo stesso nodo viene raggiunto da più archi con alta probabilità.
        In un VRP tipico, ogni cliente deve essere visitato esattamente una volta.
        """
        edge_probs = torch.sigmoid(edge_preds)

        # Per ogni nodo di destinazione, calcola la somma delle probabilità in ingresso
        # Escludiamo il depot (nodo 0) che può essere visitato più volte
        penalty = torch.tensor(0.0, device=edge_preds.device)

        for node in range(1, num_nodes):  # Salta il depot (nodo 0)
            # Trova tutti gli archi che entrano in questo nodo
            incoming_mask = edge_index[1] == node
            if incoming_mask.any():
                incoming_probs = edge_probs[incoming_mask]
                # Penalizza se la somma è > 1 (significa che più archi entrano)
                total_incoming = incoming_probs.sum()
                # Penalità quadratica per violazioni
                penalty += F.relu(total_incoming - 1.0) ** 2

        # Normalizza per il numero di nodi
        return penalty / max(num_nodes - 1, 1)

    def _capacity_penalty_v2(self, edge_preds, edge_index, node_features, capacity):
        """
        Versione migliorata della capacity penalty.
        Penalizza quando la somma delle domande lungo archi con alta probabilità supera la capacità.

        Approccio semplificato e vettorizzato per gestire batch correttamente.
        """
        if capacity is None:
            return torch.tensor(0.0, device=edge_preds.device)

        edge_probs = torch.sigmoid(edge_preds)

        # Estrai demands (assumendo indice 2 in node_features)
        demands = node_features[:, 2]

        # Per ogni arco, calcola una stima del "peso" trasportato
        # Peso = probabilità dell'arco * demand del nodo di destinazione
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # Calcola il "load" trasportato da ogni arco
        edge_loads = edge_probs * demands[dst_nodes]

        # Penalità: per ogni nodo non-depot, somma i load in ingresso
        # Se la somma supera la capacity media, c'è una violazione
        penalty = torch.tensor(0.0, device=edge_preds.device)

        # Usa la capacity media se è un batch, altrimenti scalare
        if isinstance(capacity, torch.Tensor) and capacity.numel() > 1:
            cap_value = capacity.float().mean().item()
        elif isinstance(capacity, torch.Tensor):
            cap_value = capacity.float().item()
        else:
            cap_value = float(capacity)

        # Per ogni arco dal depot, controlla se il carico stimato è ragionevole
        depot_mask = src_nodes == 0
        non_depot_dst = dst_nodes != 0

        # Archi rilevanti: depot -> customer
        relevant_mask = depot_mask & non_depot_dst

        if relevant_mask.any():
            # Per ogni cliente, somma i carichi in ingresso dal depot
            # Se supera la capacity, penalizza
            unique_customers = dst_nodes[relevant_mask].unique()

            for customer in unique_customers:
                # Trova tutti gli archi che vanno a questo cliente
                incoming_mask = (dst_nodes == customer) & (src_nodes != customer)
                if incoming_mask.any():
                    # Somma i carichi in ingresso
                    total_incoming_load = edge_loads[incoming_mask].sum()

                    # Penalizza se supera la capacity (ora cap_value è sempre scalare)
                    violation = F.relu(total_incoming_load - cap_value)
                    penalty += violation ** 2

            # Normalizza per numero di clienti
            if len(unique_customers) > 0:
                penalty = penalty / len(unique_customers)

        return penalty

    def _capacity_penalty_old(self, edge_preds, edge_index, node_features, capacity):
        """
        Penalizza percorsi che violano la capacità del veicolo.
        Usa le probabilità degli archi per stimare i percorsi e calcolare il carico.

        Args:
            edge_preds: Predizioni degli archi (logits)
            edge_index: Indici degli archi [2, num_edges]
            node_features: Features dei nodi [num_nodes, num_features]
            capacity: Capacità del veicolo (scalare o None)
        """
        if capacity is None:
            return torch.tensor(0.0, device=edge_preds.device)

        edge_probs = torch.sigmoid(edge_preds)

        # Estrai le domande dai node_features
        # Assumiamo che la domanda sia nella terza colonna (indice 2)
        # [x_coord, y_coord, demand, is_depot]
        demands = node_features[:, 2]  # shape: [num_nodes]

        # Per ogni arco con alta probabilità, calcola la domanda accumulata
        # Approssimazione: per ogni arco (i,j) con prob alta, il veicolo trasporta demand[j]
        penalty = torch.tensor(0.0, device=edge_preds.device)

        # Trova depot (nodo 0, con is_depot=1)
        depot_idx = 0

        # Per ogni possibile percorso, stimiamo il carico
        # Consideriamo solo archi con probabilità > 0.3
        high_prob_mask = edge_probs > 0.3

        if high_prob_mask.any():
            # Raggruppa archi per percorsi che partono dal depot
            for node in range(1, len(demands)):
                # Trova il percorso che collega depot -> node -> ... -> depot
                # Approssimazione: sommiamo le domande dei nodi raggiunti con alta prob
                outgoing_from_depot = edge_index[0] == depot_idx
                reaching_node = edge_index[1] == node
                relevant_edges = outgoing_from_depot | reaching_node

                if relevant_edges.any():
                    relevant_probs = edge_probs[relevant_edges]
                    relevant_dests = edge_index[1][relevant_edges]

                    # Stima del carico: somma pesata delle domande
                    estimated_load = (relevant_probs * demands[relevant_dests]).sum()

                    # Penalizza se supera la capacità
                    if isinstance(capacity, torch.Tensor):
                        cap = capacity.item() if capacity.numel() == 1 else capacity[0]
                    else:
                        cap = capacity

                    penalty += F.relu(estimated_load - cap) ** 2

        # Normalizza
        return penalty / max(len(demands) - 1, 1)

    def _route_validity_penalty(self, edge_preds, edge_index, num_nodes):
        """
        Penalizza configurazioni di percorsi non valide:
        - Ogni nodo (eccetto depot) deve avere al massimo un arco in uscita
        - Ogni nodo (eccetto depot) deve avere al massimo un arco in entrata
        - Il depot deve avere bilanciamento in/out
        """
        edge_probs = torch.sigmoid(edge_preds)
        penalty = torch.tensor(0.0, device=edge_preds.device)

        # 1. Ogni nodo non-depot deve avere al massimo un arco in uscita con alta prob
        for node in range(1, num_nodes):  # Salta depot
            outgoing_mask = edge_index[0] == node
            if outgoing_mask.any():
                outgoing_probs = edge_probs[outgoing_mask]
                total_outgoing = outgoing_probs.sum()
                # Penalizza se > 1
                penalty += F.relu(total_outgoing - 1.0) ** 2

        # 2. Bilancia depot: numero di archi in uscita ≈ numero di archi in entrata
        depot_out_mask = edge_index[0] == 0
        depot_in_mask = edge_index[1] == 0

        if depot_out_mask.any() and depot_in_mask.any():
            depot_out_sum = edge_probs[depot_out_mask].sum()
            depot_in_sum = edge_probs[depot_in_mask].sum()

            # Penalizza differenza (dovrebbero essere uguali)
            penalty += (depot_out_sum - depot_in_sum) ** 2

        return penalty / max(num_nodes - 1, 1)


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