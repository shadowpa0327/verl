"""
ActorRolloutRefDrafterWorker — extends ActorRolloutRefWorker with EAGLE drafter co-training.

Adds:
- self.hs_collector: VllmHSCollector (vLLM with KV connector, colocated, sleep/wake)
- self.drafter: TrainingWorker (FSDPDrafterEngine for EAGLE model training)

The DrafterDataController lives on the driver (RayPPOTrainer), not here.
This worker only:
- Receives per-rank data via mesh dispatch (update_drafter)
- Fetches tensors from Mooncake
- Runs drafter training
- Participates in HS collection when asked

See claude_docs/rfc-drafter-trainer-integration.md for the full design.
"""

import logging
from typing import Optional

import torch
from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.engine_workers import ActorRolloutRefWorker

logger = logging.getLogger(__name__)


class ActorRolloutRefDrafterWorker(ActorRolloutRefWorker):
    """Extends ActorRolloutRefWorker with EAGLE drafter co-training.

    Worker hierarchy:
        self.actor           (TrainingWorker → FSDPEngine)       [inherited]
        self.ref             (TrainingWorker → FSDPEngine)       [inherited]
        self.rollout         (BaseRollout → vLLM)                [inherited]
        self.hs_collector    (VllmHSCollector → vLLM with KV connector) [NEW]
        self.drafter         (TrainingWorker → FSDPDrafterEngine)       [NEW]

    GPU time-multiplexing order per RL step:
        rollout (AWAKE) → hs_collector (AWAKE) → drafter (AWAKE) → actor (AWAKE)
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        self.hs_collector = None
        self.drafter = None
        self._drafter_enabled = config.get("drafter", {}).get("enable", False)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # 1-4. Actor, ref, rollout, checkpoint — all inherited
        super().init_model()

        if not self._drafter_enabled:
            return

        # 5. Build HS collector (colocated vLLM with KV connector)
        self._init_hs_collector()

        # 6. Build drafter training engine
        self._init_drafter()

        # 7. Register drafter mesh (pure DP — every rank is unique)
        import torch.distributed as dist
        self._register_dispatch_collect_info(
            mesh_name="drafter",
            dp_rank=dist.get_rank(),  # world_rank = dp_rank (pure DP)
            is_collect=True,
        )

        logger.info("Drafter co-training initialized (HS collector + drafter engine)")

    def _init_hs_collector(self):
        """Initialize the vLLM HS collector.

        Same GPU as rollout, time-multiplexed via sleep/wake.
        Uses vLLM with extract_hidden_states speculative config + KV connector.
        """
        from verl.workers.rollout.vllm_rollout.vllm_hs_collector import VllmHSCollector

        hs_config = self.config.get("hs_collector", {})
        mooncake_config = self.config.get("mooncake", None)

        self.hs_collector = VllmHSCollector(
            args=hs_config,
            mooncake_config=mooncake_config,
        )

        logger.info("vLLM HS collector initialized")

    def _init_drafter(self):
        """Initialize the drafter training engine.

        Uses FSDPDrafterEngine (registered as model_type="drafter_model").
        Shares embed_tokens/lm_head from actor (frozen, zero copy).
        """
        from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig

        drafter_cfg = self.config.drafter
        drafter_training_config = TrainingWorkerConfig(
            model_type="drafter_model",
            model_config=drafter_cfg.get("model_config", {}),
            engine_config=drafter_cfg.get("engine_config", {}),
            optimizer_config=drafter_cfg.get("optimizer_config", {}),
        )
        self.drafter = TrainingWorker(config=drafter_training_config)
        self.drafter.reset()

        # Set custom loss function (Forward KL for EAGLE)
        if drafter_cfg.get("loss_fn", None) == "eagle_forward_kl":
            from verl.models.eagle3.ops.loss import compiled_forward_kl_loss
            self.drafter.set_loss_fn(compiled_forward_kl_loss)

        # Initial sync of frozen modules (embed_tokens, lm_head) from actor.
        # These are weight copies, not references — must be re-synced after
        # each update_actor() since the actor trains every RL step.
        self._sync_drafter_frozen_modules()

        logger.info("Drafter TrainingWorker initialized")

    def _sync_drafter_frozen_modules(self):
        """Copy frozen weights from actor into drafter.

        Syncs: embed_tokens, target_lm_head_weight, verifier_norm (final RMSNorm).
        All frozen (requires_grad=False). embed_tokens for drafter input,
        target_lm_head_weight for target distribution, verifier_norm for
        pre-norm correction. The drafter's own lm_head is trainable and NOT synced.

        Called at init and after each update_actor(). In verl the actor trains
        every RL step (unlike TorchSpec where the target is fixed), so the
        drafter's frozen copies must stay synchronized.
        """
        if self.actor is None or self.drafter is None:
            return
        if not hasattr(self.drafter.engine, "sync_frozen_modules_from_actor"):
            return

        actor_module = self.actor.engine.module

        # Get actor's final norm (model.norm — the RMSNorm before lm_head).
        # This is the "verifier_norm" needed because vLLM captures
        # last_hidden_states pre-norm.
        actor_norm = None
        if hasattr(actor_module, "model") and hasattr(actor_module.model, "norm"):
            actor_norm = actor_module.model.norm

        self.drafter.engine.sync_frozen_modules_from_actor(
            actor_embed_tokens=actor_module.model.embed_tokens,
            actor_lm_head=actor_module.lm_head,
            actor_norm=actor_norm,
        )

    # ── HS Collection ─────────────────────────────────────────

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def collect_hidden_states(self, sequences: DataProto) -> DataProto:
        """Run HS collection: rollout sleeps → HS collector wakes → prefill → Mooncake → metadata.

        Called by driver after generate_sequences().
        Returns DataProto with SampleMeta packed in non_tensor_batch.
        """
        if self.hs_collector is None:
            return DataProto()

        # TODO: Sleep/wake coordination
        # self.rollout.release()  — rollout should already be sleeping after generate
        # self.hs_collector.resume(["weights"])

        sample_metas = self.hs_collector.collect_hidden_states(sequences)

        # self.hs_collector.release()

        # Pack SampleMeta list into DataProto for driver collection
        import numpy as np
        if not sample_metas:
            return DataProto()

        return DataProto(
            non_tensor_batch={
                'mooncake_keys': np.array([m.mooncake_key for m in sample_metas], dtype=object),
                'shapes': np.array([m.shapes for m in sample_metas], dtype=object),
                'dtypes': np.array([m.dtypes for m in sample_metas], dtype=object),
                'seq_lens': np.array([m.seq_len for m in sample_metas], dtype=object),
                'n_tokens': np.array([m.n_tokens for m in sample_metas], dtype=object),
            },
        )

    # ── Drafter Training ──────────────────────────────────────

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter"))
    def update_drafter(self, data: DataProto):
        """Train drafter on this rank's shard of hidden states data.

        data is already split per DP rank by the drafter mesh dispatch fn.
        Each entry contains Mooncake keys — actual tensors fetched at train time.

        Dispatch: make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter")
        splits DataProto.non_tensor_batch per rank via np.array_split.
        """
        if self.drafter is None or len(data) == 0:
            return

        drafter_cfg = self.config.drafter
        max_steps = drafter_cfg.get("max_steps", 1)
        batch_size = drafter_cfg.get("batch_size", len(data))

        mooncake_keys = data.non_tensor_batch.get('mooncake_keys', [])
        shapes_list = data.non_tensor_batch.get('shapes', [])
        dtypes_list = data.non_tensor_batch.get('dtypes', [])

        if len(mooncake_keys) == 0:
            return

        # Fetch tensors from Mooncake and train
        from verl.utils.mooncake import EagleMooncakeStore

        for step in range(max_steps):
            start = step * batch_size
            end = min(start + batch_size, len(mooncake_keys))
            if start >= len(mooncake_keys):
                break

            batch_keys = mooncake_keys[start:end]
            batch_shapes = shapes_list[start:end]
            batch_dtypes = dtypes_list[start:end]

            # TODO: Fetch from Mooncake and run train_batch
            # For each key in batch_keys:
            #   tensors = mooncake_store.get(key, shapes, dtypes, device)
            #   self.drafter.train_batch(data=tensors)
            #   mooncake_store.remove_eagle3_tensors(key)

            logger.debug("update_drafter: step %d, keys %d-%d", step, start, end)

    # ── Weight Sync ───────────────────────────────────────────

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        """Sync actor + drafter weights to rollout + HS collector.

        Extended from parent to also:
        1. Re-sync frozen modules (embed_tokens, lm_head) from actor → drafter
           (actor weights changed during update_actor())
        2. Sync actor weights to HS collector (same weights as rollout)
        3. Sync drafter weights to rollout (for speculative decoding)
        """
        # Actor → rollout (inherited)
        await super().update_weights()

        # Actor → drafter frozen modules (embed_tokens, lm_head changed after training)
        if self.drafter is not None:
            self._sync_drafter_frozen_modules()

        if self.hs_collector is not None:
            # Actor → HS collector (same weights, separate server)
            per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param()
            await self.hs_collector.update_weights(per_tensor_param, peft_config=peft_config)

        if self.drafter is not None and self.rollout is not None:
            # Drafter → rollout (for speculative decoding at inference time)
            drafter_params, _ = self.drafter.engine.get_per_tensor_param()
            # TODO: self.rollout.update_drafter_weights(drafter_params)
            # Requires rollout to support drafter weight updates
            pass
