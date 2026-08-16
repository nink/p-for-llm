from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 32_768
    d_model: int = 192
    n_layers: int = 12
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_hidden: int = 512
    n_experts: int = 29
    ple_dim: int = 176
    max_seq_len: int = 1_024
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    router_balance_loss_coefficient: float = 0.01
    router_z_loss_coefficient: float = 0.001
    router_top_k: int = 1

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if min(
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.ffn_hidden,
            self.n_experts,
            self.ple_dim,
            self.max_seq_len,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if min(
            self.router_balance_loss_coefficient,
            self.router_z_loss_coefficient,
        ) < 0.0:
            raise ValueError("router loss coefficients must be non-negative")
        if not 1 <= self.router_top_k <= self.n_experts:
            raise ValueError("router_top_k must be between 1 and n_experts")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def tiny(cls, vocab_size: int = 256) -> "ModelConfig":
        return cls(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            ffn_hidden=128,
            n_experts=4,
            ple_dim=16,
            max_seq_len=64,
        )
