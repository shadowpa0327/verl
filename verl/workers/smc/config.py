from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SMCConfig:
    """Configuration for SMC decoding."""

    mode: str = "smc"  # smc | sd_like
    n_particles: int = 4
    gamma: int = 8
    draft_temperature: float = 0.7
    target_lhts_temperature: float = 1.0
    resample_threshold: float = 0.5
    rejuvenate: bool = False
    rejuvenation_k: int = 16
    rejuvenation_frac: float = 0.5
    rejuvenation_logp_threshold: float = -3.0
    rejuvenation_logp_ema_alpha: float = 0.3
    rejuvenation_trigger_steps: int = 2
    resample_method: str = "systematic"  # systematic | multinomial


@dataclass
class MultiGPUSMCConfig:
    """Configuration for multi-GPU SMC speculative decoding."""

    smc: SMCConfig = field(default_factory=SMCConfig)
    draft_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    target_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    n_gpus: int = 4
    target_tp_size: int = 4
    draft_mem_fraction: float = 0.25
    target_mem_fraction: float = 0.55
    draft_quantization: Optional[str] = None
    target_quantization: Optional[str] = None

    @property
    def particles_per_gpu(self) -> int:
        """Number of particles each draft worker handles."""
        n = self.smc.n_particles
        k = self.n_gpus
        # Round up so all particles are covered
        return (n + k - 1) // k
