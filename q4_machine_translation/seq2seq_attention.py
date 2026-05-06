"""
Seq2Seq model with attention for machine translation (Q4).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, SEED

torch.manual_seed(SEED)


class Encoder(nn.Module):
    """Encoder for Seq2Seq model."""
    
    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.5):
        """
        Initialize encoder.
        
        Args:
            input_dim: Vocabulary size (source)
            emb_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers,
            dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            src: Source sequence [batch, src_len]
            
        Returns:
            outputs, (hidden, cell)
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    """Attention mechanism."""
    
    def __init__(self, hidden_dim: int):
        """
        Initialize attention.
        
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden: torch.Tensor, 
                encoder_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute attention weights.
        
        Args:
            hidden: Decoder hidden state [batch, hidden_dim]
            encoder_outputs: Encoder outputs [batch, src_len, hidden_dim]
            mask: Mask for padded positions
            
        Returns:
            Attention weights [batch, src_len]
        """
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden state for each source position
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Concatenate and compute energy
        energy = torch.tanh(self.attention(
            torch.cat((hidden, encoder_outputs), dim=2)
        ))
        
        attention = self.v(energy).squeeze(2)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """Decoder with attention."""
    
    def __init__(self, output_dim: int, emb_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.5):
        """
        Initialize decoder.
        
        Args:
            output_dim: Vocabulary size (target)
            emb_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.output_dim = output_dim
        self.attention = Attention(hidden_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim + hidden_dim, hidden_dim, num_layers,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(
            hidden_dim * 2 + emb_dim, output_dim
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor, hidden: torch.Tensor,
                cell: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for one decoding step.
        
        Args:
            input: Input token [batch, 1]
            hidden: Previous hidden state
            cell: Previous cell state
            encoder_outputs: Encoder outputs
            mask: Source mask
            
        Returns:
            prediction, hidden, cell, attention_weights
        """
        embedded = self.dropout(self.embedding(input))
        
        # Compute attention
        a = self.attention(hidden[-1], encoder_outputs, mask)
        a = a.unsqueeze(1)
        
        # Weighted sum of encoder outputs
        weighted = torch.bmm(a, encoder_outputs)
        
        # LSTM input
        lstm_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Prediction
        prediction = self.fc_out(torch.cat((
            output, weighted, embedded
        ), dim=2))
        
        return prediction, hidden, cell, a.squeeze(1)


class Seq2SeqAttention(nn.Module):
    """
    Complete Seq2Seq model with attention.
    """
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        """
        Initialize Seq2Seq model.
        
        Args:
            encoder: Encoder module
            decoder: Decoder module
            device: Device to use
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequence [batch, src_len]
            tgt: Target sequence [batch, tgt_len]
            teacher_forcing_ratio: Probability of teacher forcing
            
        Returns:
            Output predictions [batch, tgt_len, output_dim]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        output_dim = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, tgt_len, output_dim).to(self.device)
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First input is SOS token
        input = tgt[:, 0].unsqueeze(1)
        
        # Create mask for source padding
        mask = (src != 0).to(self.device)
        
        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(
                input, hidden, cell, encoder_outputs, mask
            )
            
            outputs[:, t, :] = output.squeeze(1)
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs
    
    def translate(self, src: torch.Tensor, max_len: int = 50,
                  sos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        """
        Translate source sequence.
        
        Args:
            src: Source sequence
            max_len: Maximum translation length
            sos_idx: Start of sentence index
            eos_idx: End of sentence index
            
        Returns:
            Translated sequence
        """
        self.eval()
        
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)
            mask = (src != 0).to(self.device)
            
            # Start with SOS token
            input = torch.tensor([[sos_idx]] * src.shape[0]).to(self.device)
            
            translations = []
            for _ in range(max_len):
                output, hidden, cell, _ = self.decoder(
                    input, hidden, cell, encoder_outputs, mask
                )
                
                pred = output.argmax(2)
                translations.append(pred)
                
                input = pred
        
        return torch.cat(translations, dim=1)