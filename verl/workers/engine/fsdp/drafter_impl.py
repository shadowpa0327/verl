"""
FSDPDrafterEngine — FSDP engine for EAGLE drafter model training.

Registered as model_type="drafter_model" in the EngineRegistry.
Wraps the EAGLE draft model (fc + 1 decoder layer, ~2% of target params)
with FSDP for distributed training.

Key differences from FSDPEngineWithLMHead:
- Loads EAGLE draft model (not the full target model)
- Borrows embed_tokens/lm_head from actor (frozen, shared refs)
- Only FSDP-wraps the trainable params (fc + decoder layer)
- Uses Forward KL loss (not cross-entropy)
- prepare_model_inputs() expects hidden_states (from Mooncake) + input_ids

Usage:
    engine = EngineRegistry.new("drafter_model", "fsdp", "cuda", ...)
    engine.set_shared_modules(embed_tokens=..., lm_head=...)
"""

import logging
from typing import Optional

import torch
from tensordict import TensorDict

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

logger = logging.getLogger(__name__)


@EngineRegistry.register(model_type="drafter_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPDrafterEngine(FSDPEngine):
    """
    FSDP engine for EAGLE drafter model.

    Tiny model (~2% of target params): fc projection + 1 decoder layer.
    Borrows embed_tokens/lm_head from the actor model (frozen, shared refs).
    Trained with Forward KL loss on hidden states extracted by the HS collector.
    """

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        self._shared_embed_tokens = None
        self._shared_lm_head = None

    def initialize(self):
        """Load the EAGLE draft model and set up FSDP.

        The draft model is loaded from the checkpoint specified in model_config.
        After loading, call set_shared_modules() to link embed_tokens/lm_head
        from the actor model.
        """
        # Load draft model using AutoEagle3DraftModel
        from verl.models.eagle3.draft.auto import AutoEagle3DraftModel, AutoDraftModelConfig

        draft_config = AutoDraftModelConfig.from_file(self.model_config.local_path)
        self.module = AutoEagle3DraftModel.from_config(
            draft_config,
            torch_dtype=getattr(torch, self.model_config.dtype, torch.bfloat16),
        )

        # Freeze embed_tokens if present (will be replaced by shared ref)
        if hasattr(self.module, "freeze_embedding"):
            self.module.freeze_embedding()

        logger.info(
            "Draft model loaded: %s params (%.1fM)",
            sum(p.numel() for p in self.module.parameters()),
            sum(p.numel() for p in self.module.parameters()) / 1e6,
        )

        # Set up FSDP, optimizer, scheduler via parent
        super().initialize()

    def set_shared_modules(
        self,
        embed_tokens: torch.nn.Module,
        lm_head: torch.nn.Module,
    ):
        """Set shared frozen modules from the target (actor) model.

        These are reference-shared (zero copy). The drafter uses them
        for embedding and final projection but does not update their gradients.

        Args:
            embed_tokens: Actor model's embedding layer (frozen)
            lm_head: Actor model's output projection (frozen)
        """
        self._shared_embed_tokens = embed_tokens
        self._shared_lm_head = lm_head

        # If the draft model has load_embedding / set methods, use them
        if hasattr(self.module, "model") and hasattr(self.module.model, "embed_tokens"):
            self.module.model.embed_tokens = embed_tokens
        if hasattr(self.module, "lm_head"):
            self.module.lm_head = lm_head

        # Freeze shared params
        for param in embed_tokens.parameters():
            param.requires_grad = False
        for param in lm_head.parameters():
            param.requires_grad = False

        logger.info("Shared embed_tokens and lm_head from actor (frozen)")

    def prepare_model_inputs(self, micro_batch: TensorDict):
        """Prepare inputs for the EAGLE draft model forward pass.

        Expects:
            micro_batch["input_ids"]: Token IDs [batch, seq]
            micro_batch["hidden_states"]: Hidden states from target model [batch, seq, hidden_dim]
            micro_batch["attention_mask"]: Attention mask [batch, seq] (optional)

        The hidden states come from Mooncake (fetched by the drafter worker
        before calling train_batch).
        """
        input_ids = micro_batch.get("input_ids", None)
        hidden_states = micro_batch.get("hidden_states", None)
        attention_mask = micro_batch.get("attention_mask", None)

        model_inputs = {
            "input_ids": input_ids,
            "hidden_states": hidden_states,
        }
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        return model_inputs
