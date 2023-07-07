# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import datetime
import functools
import os
from datetime import timedelta
from re import escape
from unittest import mock
from unittest.mock import ANY, MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from torch.optim import Adam

from lightning.fabric import Fabric
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.strategies.fsdp import _FSDPBackwardSyncControl, fsdp_overlap_step_with_backward
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12, _TORCH_GREATER_EQUAL_2_1
from tests_fabric.helpers.runif import RunIf
from tests_fabric.strategies.test_single_device import _MyFabricGradNorm

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel, MixedPrecision


@mock.patch("lightning.fabric.strategies.fsdp._TORCH_GREATER_EQUAL_1_12", False)
def test_fsdp_support(*_):
    with pytest.raises(NotImplementedError, match="`FSDPStrategy` is supported from PyTorch v1.12.0"):
        FSDPStrategy()


@RunIf(min_torch="1.12")
def test_fsdp_custom_mixed_precision():
    """Test that passing a custom mixed precision config works."""
    config = MixedPrecision()
    strategy = FSDPStrategy(mixed_precision=config)
    assert strategy.mixed_precision_config == config


@RunIf(min_torch="1.12")
def test_fsdp_cpu_offload():
    """Test the different ways cpu offloading can be enabled."""
    # bool
    strategy = FSDPStrategy(cpu_offload=True)
    assert strategy.cpu_offload == CPUOffload(offload_params=True)

    # dataclass
    config = CPUOffload()
    strategy = FSDPStrategy(cpu_offload=config)
    assert strategy.cpu_offload == config


@RunIf(min_torch="1.12")
@pytest.mark.parametrize("torch_ge_2_0", [False, True])
def test_fsdp_setup_optimizer_validation(torch_ge_2_0):
    """Test that `setup_optimizer()` validates the param groups and reference to FSDP parameters."""
    module = nn.Linear(2, 2)
    strategy = FSDPStrategy(parallel_devices=[torch.device("cpu")])

    with mock.patch("lightning.fabric.strategies.fsdp._TORCH_GREATER_EQUAL_2_0", torch_ge_2_0):
        bad_optimizer_1 = Adam([{"params": [module.weight]}, {"params": [module.bias], "lr": 1e-3}])
        bad_optimizer_2 = Adam(module.parameters())

        if torch_ge_2_0:
            strategy.setup_optimizer(bad_optimizer_1)
            strategy.setup_optimizer(bad_optimizer_2)
        else:
            with pytest.raises(ValueError, match="does not support multiple param groups"):
                strategy.setup_optimizer(bad_optimizer_1)
            with pytest.raises(ValueError, match="The optimizer does not seem to reference any FSDP parameter"):
                strategy.setup_optimizer(bad_optimizer_2)


@RunIf(min_torch="2.0.0")
@mock.patch("lightning.fabric.strategies.fsdp.FSDPStrategy.setup_module")
def test_fsdp_setup_use_orig_params(_):
    module = nn.Linear(2, 2)
    optimizer = Adam(module.parameters())

    strategy = FSDPStrategy(parallel_devices=[torch.device("cpu")], use_orig_params=False)
    assert not strategy._fsdp_kwargs["use_orig_params"]

    with pytest.raises(ValueError, match=r"`FSDPStrategy\(use_orig_params=False\)` but this is not supported"):
        strategy.setup_module_and_optimizers(module, optimizer)

    strategy = FSDPStrategy(parallel_devices=[torch.device("cpu")])
    assert strategy._fsdp_kwargs["use_orig_params"]
    strategy.setup_module_and_optimizers(module, optimizer)
    assert strategy._fsdp_kwargs["use_orig_params"]


@RunIf(min_torch="1.12")
def test_fsdp_no_backward_sync():
    """Test that the backward sync control calls `.no_sync()`, and only on a module wrapped in
    FullyShardedDataParallel."""

    strategy = FSDPStrategy()
    assert isinstance(strategy._backward_sync_control, _FSDPBackwardSyncControl)

    with pytest.raises(
        TypeError, match="is only possible if the module passed to .* is wrapped in `FullyShardedDataParallel`"
    ), strategy._backward_sync_control.no_backward_sync(Mock()):
        pass

    module = MagicMock(spec=FullyShardedDataParallel)
    with strategy._backward_sync_control.no_backward_sync(module):
        pass

    module.no_sync.assert_called_once()


@RunIf(min_torch="1.12")
@mock.patch("lightning.fabric.strategies.fsdp._TORCH_GREATER_EQUAL_1_13", False)
def test_fsdp_activation_checkpointing_support():
    """Test that we error out if activation checkpointing requires a newer PyTorch version."""
    with pytest.raises(ValueError, match="activation_checkpointing` requires torch >= 1.13.0"):
        FSDPStrategy(activation_checkpointing=Mock())


@RunIf(min_torch="1.13")
def test_fsdp_activation_checkpointing():
    """Test that the FSDP strategy can apply activation checkpointing to the given layers."""

    class Block1(nn.Linear):
        pass

    class Block2(nn.Linear):
        pass

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Sequential(Block1(4, 4), Block1(5, 5))
            self.layer1 = Block2(2, 2)
            self.layer2 = nn.Linear(3, 3)

    if _TORCH_GREATER_EQUAL_2_1:
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        strategy = FSDPStrategy(activation_checkpointing_policy=ModuleWrapPolicy({Block1}))
        assert set(strategy._activation_checkpointing_kwargs) == {"auto_wrap_policy"}

        strategy = FSDPStrategy(activation_checkpointing_policy=ModuleWrapPolicy({Block1, Block2}))
        assert set(strategy._activation_checkpointing_kwargs) == {"auto_wrap_policy"}
    else:
        strategy = FSDPStrategy(activation_checkpointing=Block1)
        assert set(strategy._activation_checkpointing_kwargs) == {"check_fn"}

        strategy = FSDPStrategy(activation_checkpointing=[Block1, Block2])
        assert set(strategy._activation_checkpointing_kwargs) == {"check_fn"}

    strategy._parallel_devices = [torch.device("cuda", 0)]
    with mock.patch(
        "torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel"
    ) as fsdp_mock, mock.patch(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing"
    ) as ckpt_mock:
        strategy.setup_module(Model())
        ckpt_mock.assert_called_with(
            fsdp_mock(), checkpoint_wrapper_fn=ANY, **strategy._activation_checkpointing_kwargs
        )


@RunIf(min_torch="1.13", min_cuda_gpus=1)
def test_fsdp_activation_checkpointing_manual():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))

    if _TORCH_GREATER_EQUAL_2_1:
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        strategy = FSDPStrategy(activation_checkpointing_policy=ModuleWrapPolicy({torch.nn.Linear}))
    else:
        strategy = FSDPStrategy(activation_checkpointing=torch.nn.Linear)

    fabric = Fabric(devices=1, accelerator="cuda", strategy=strategy)
    fabric.launch()

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        CheckpointWrapper,
    )

    # manually apply activation checkpointing
    apply_activation_checkpointing(model)

    wrappers = {name for name, mod in model.named_modules() if isinstance(mod, CheckpointWrapper)}
    assert wrappers == {"0", "1"}

    # let fabric set-up the model, it shouldn't apply activation checkpointing again
    with pytest.warns(match="is configured, but the model already contains checkpointed"):
        model = fabric.setup(model)

    wrappers = {name for name, mod in model._forward_module.named_modules() if isinstance(mod, CheckpointWrapper)}
    assert wrappers == {"_fsdp_wrapped_module.0", "_fsdp_wrapped_module.1"}


@RunIf(min_torch="1.13")
def test_fsdp_grad_clipping_value_error():
    strategy = FSDPStrategy()
    with pytest.raises(
        NotImplementedError,
        match=(
            "FSDP currently does not support to clip gradients by value. "
            "Consider clipping by norm instead or choose another strategy!"
        ),
    ):
        strategy.clip_gradients_value(Mock(), Mock(), Mock())


class _MyFSDPFabricGradientNorm(_MyFabricGradNorm):
    def after_backward(self, model, optimizer):
        self.clip_gradients(model, optimizer, max_norm=0.05, error_if_nonfinite=True)

        with model._forward_module.summon_full_params(model._forward_module):
            parameters = model.parameters()
            grad_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad.detach(), 2, dtype=torch.float32) for p in parameters]),
                2,
            )
            torch.testing.assert_close(grad_norm, torch.tensor(0.05, device=self.device))


@pytest.mark.parametrize(
    "precision",
    ["32-true", "16-mixed", pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True))],
)
@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.xfail(reason="Testing with FSDP is not yet correct")  # TODO: Investigate testing with fsdp
def test_fsdp_grad_clipping_norm(precision):
    fabric = _MyFSDPFabricGradientNorm(accelerator="cuda", devices=2, precision=precision, strategy="fsdp")
    fabric.run()


@RunIf(min_torch="2.0.0")
def test_fsdp_save_checkpoint_storage_options(tmp_path):
    """Test that the FSDP strategy does not accept storage options for saving checkpoints."""
    strategy = FSDPStrategy()
    with pytest.raises(TypeError, match=escape("FSDPStrategy.save_checkpoint(..., storage_options=...)` is not")):
        strategy.save_checkpoint(path=tmp_path, state=Mock(), storage_options=Mock())


@RunIf(min_torch="2.0.0")
@mock.patch("lightning.fabric.strategies.fsdp.FSDPStrategy.broadcast", lambda _, x: x)
def test_fsdp_save_checkpoint_folder_exists(tmp_path):
    path = tmp_path / "exists"
    path.mkdir()
    (path / "file").touch()
    strategy = FSDPStrategy()
    with pytest.raises(FileExistsError, match="exists and is not empty"):
        strategy.save_checkpoint(path=path, state=Mock())


@RunIf(min_torch="2.0.0")
@mock.patch("lightning.fabric.strategies.fsdp.FSDPStrategy.broadcast", lambda _, x: x)
def test_fsdp_save_checkpoint_one_fsdp_module_required(tmp_path):
    """Test that the FSDP strategy can only save one FSDP model per checkpoint."""
    strategy = FSDPStrategy()

    # missing FSDP model
    with pytest.raises(ValueError, match="Could not find a FSDP model in the provided checkpoint state."):
        strategy.save_checkpoint(path=tmp_path, state={})
    with pytest.raises(ValueError, match="Could not find a FSDP model in the provided checkpoint state."):
        strategy.save_checkpoint(path=tmp_path, state={"model": torch.nn.Linear(3, 3)})

    # multiple FSDP models
    model1 = Mock(spec=FullyShardedDataParallel)
    model2 = Mock(spec=FullyShardedDataParallel)
    with pytest.raises(ValueError, match="Found multiple FSDP modules in the given state."):
        strategy.save_checkpoint(path=tmp_path, state={"model1": model1, "model2": model2})


@RunIf(min_torch="2.0.0")
def test_fsdp_load_checkpoint_no_state(tmp_path):
    """Test that the FSDP strategy can't load the full state without access to a model instance from the user."""
    strategy = FSDPStrategy()
    with pytest.raises(ValueError, match=escape("Got FSDPStrategy.load_checkpoint(..., state=None")):
        strategy.load_checkpoint(path=tmp_path, state=None)
    with pytest.raises(ValueError, match=escape("Got FSDPStrategy.load_checkpoint(..., state={})")):
        strategy.load_checkpoint(path=tmp_path, state={})


@RunIf(min_torch="2.0.0")
@mock.patch("lightning.fabric.strategies.fsdp.FSDPStrategy.broadcast", lambda _, x: x)
def test_fsdp_load_checkpoint_one_fsdp_module_required(tmp_path):
    """Test that the FSDP strategy can only load one FSDP model per checkpoint."""
    strategy = FSDPStrategy()

    # missing FSDP model
    with pytest.raises(ValueError, match="Could not find a FSDP model in the provided checkpoint state."):
        strategy.load_checkpoint(path=tmp_path, state={"other": "data"})
    with pytest.raises(ValueError, match="Could not find a FSDP model in the provided checkpoint state."):
        strategy.load_checkpoint(path=tmp_path, state={"model": torch.nn.Linear(3, 3)})

    # multiple FSDP models
    model1 = Mock(spec=FullyShardedDataParallel)
    model2 = Mock(spec=FullyShardedDataParallel)
    with pytest.raises(ValueError, match="Found multiple FSDP modules in the given state."):
        strategy.load_checkpoint(path=tmp_path, state={"model1": model1, "model2": model2})


@RunIf(min_torch="2.0.0")
@mock.patch("lightning.fabric.strategies.fsdp.FSDPStrategy.broadcast", lambda _, x: x)
def test_fsdp_save_checkpoint_unknown_state_dict_type(tmp_path):
    strategy = FSDPStrategy(state_dict_type="invalid")
    model = Mock(spec=FullyShardedDataParallel)
    with pytest.raises(ValueError, match="Unknown state_dict_type"):
        strategy.save_checkpoint(path=tmp_path, state={"model": model})


@RunIf(min_torch="2.0.0")
def test_fsdp_load_unknown_checkpoint_type(tmp_path):
    """Test that the strategy validates the contents at the checkpoint path."""
    strategy = FSDPStrategy()
    model = Mock(spec=FullyShardedDataParallel)
    path = tmp_path / "empty_dir"  # neither a single file nor a directory with meta file
    path.mkdir()
    with pytest.raises(ValueError, match="does not point to a valid checkpoint"):
        strategy.load_checkpoint(path=path, state={"model": model})


@RunIf(min_torch="1.12")
@mock.patch("torch.distributed.init_process_group")
def test_set_timeout(init_process_group_mock):
    """Test that the timeout gets passed to the ``torch.distributed.init_process_group`` function."""
    test_timedelta = timedelta(seconds=30)
    strategy = FSDPStrategy(timeout=test_timedelta, parallel_devices=[torch.device("cpu")])
    strategy.cluster_environment = LightningEnvironment()
    strategy.accelerator = Mock()
    strategy.setup_environment()
    process_group_backend = strategy._get_process_group_backend()
    global_rank = strategy.cluster_environment.global_rank()
    world_size = strategy.cluster_environment.world_size()
    init_process_group_mock.assert_called_with(
        process_group_backend, rank=global_rank, world_size=world_size, timeout=test_timedelta
    )


class SubBlock(nn.Sequential):
    def __init__(self, feature_dim: int) -> None:
        super().__init__(
            nn.Linear(feature_dim, feature_dim, bias=False),
            torch.nn.LayerNorm([feature_dim]),
            nn.ReLU(),
        )


class Block(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.left = SubBlock(feature_dim)
        self.right = SubBlock(feature_dim)

    def forward(self, x):
        return self.left(x) + self.right(x)


class StatusChecker:
    def __init__(self, fabric: Fabric) -> None:
        self._fabric = fabric
        self.is_rank_zero = fabric.is_global_zero
        self.pids = tuple(int(pid) for pid in fabric.all_gather(os.getpid()).cpu().numpy())

    @contextlib.contextmanager
    def guard_region(self, name: str):
        """Handle errors and graceful shutdown.

        `pytest` interprets SystemExit as a faiure, so it will interpret shutdown of non-zero ranks as a test failure.
        This is confusing (since it logs "FAILED"), but more importantly the orphan rank will continue trying to execute
        the rest of the test suite. So instead we add calls to `os._exit` which actually forces the process to shut
        down.
        """
        success = False
        try:
            yield
            success = True

        except BaseException:
            if self.is_rank_zero:
                raise

        finally:
            # All reduce will wait for all workers to enter. This means that if a
            # worker dies the status check will deadlock.
            import psutil

            worker_status = tuple(psutil.Process(pid).status() for pid in self.pids)
            if any(
                status in (psutil.STATUS_DEAD, psutil.STATUS_STOPPED, psutil.STATUS_ZOMBIE) for status in worker_status
            ):
                if self.is_rank_zero:
                    raise RuntimeError(f"({name}) Dead workers: [{', '.join(worker_status)}]")
                else:
                    os._exit(1)

            rank_success = self._fabric.all_gather(success).cpu()
            if not rank_success.all():
                if self.is_rank_zero > 0:
                    os._exit(1)
                elif success:
                    raise RuntimeError(f"({name}) Failure on different rank: {rank_success}")

    def finalize(self) -> None:
        if not self.is_rank_zero:
            os._exit(0)

    def __del__(self) -> None:
        self.finalize()


@pytest.mark.skip(reason="Flaky test")  # See also: https://github.com/Lightning-AI/lightning/pull/17774
@RunIf(min_torch="2.0.0", min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.skipif(not RequirementCache("psutil"), reason="psutil is needed to help prevent deadlocks.")
@pytest.mark.parametrize(
    "checkpoint",
    [(Block,), (SubBlock,), (Block, SubBlock, nn.Linear), None],
)
def test_apply_optimizer_in_backward(checkpoint):
    try:
        from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
    except ImportError:
        pytest.skip("Failed to import `lambda_auto_wrap_policy`")

    from torch.distributed.fsdp._traversal_utils import _get_fsdp_handles

    num_gpus = 2
    num_blocks = 8
    feature_dim = 256

    # This bound is dependent on the topology of the model. The grads for each
    # Block are two `feature_dim ** 2` Tensors (`left` and `right` Linear layers)
    # times four. (FP32 = 4 bytes / element)
    #
    # In the baseline case grads for all Blocks are materialized at once, whereas
    # in the test case only one Block should have grads in memory which adds a
    # `(num_blocks - 1)` factor.
    #
    # However, there is one final correction to be made. In the baseline case peak
    # memory occurs at the end of the backward pass; at that time activations will
    # have been freed and will offset the memory relative to the base case. (Which
    # reaches peak memory after the first block when most activations are still
    # in memory.) It's difficult to estimate the exact correction factor
    # (particularly since it varies with activation checkpointing strategy), but
    # three is close enough for our purposes.
    upper_savings_bound = 4 * feature_dim**2 * 2 * (num_blocks - 1)
    lower_savings_bound = upper_savings_bound / 3

    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda module: isinstance(module, Block))
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing=checkpoint,
        timeout=datetime.timedelta(seconds=10),
    )
    fabric = Fabric(accelerator="cuda", devices=num_gpus, strategy=strategy)
    fabric.launch()
    status_checker = StatusChecker(fabric)

    def make_model_and_optimizers():
        torch.manual_seed(0)

        with fabric.init_module():
            backbone = [Block(feature_dim) for _ in range(num_blocks)]
            model = nn.Sequential(*backbone, nn.Linear(feature_dim, 1, bias=False))
            optimizers = [torch.optim.SGD(layer.parameters(), lr=0.1, momentum=0.9) for layer in model]

        return fabric.setup_module(model), fabric.setup_optimizers(*optimizers)

    with status_checker.guard_region("Instantiate model."):
        baseline_model, baseline_optimizers = make_model_and_optimizers()
        test_model, test_optimizers = make_model_and_optimizers()
        fabric.seed_everything(1337 + fabric.global_rank)

        # Check that initialization is identical.
        for p0, p1 in zip(baseline_model.parameters(), test_model.parameters()):
            assert (p0 == p1).all()

    num_steps = 50
    for step in range(num_steps):
        # Normal pattern: `.backward()` followed by `.step()`
        with status_checker.guard_region(f"({step + 1} / {num_steps}) Baseline"):
            x = torch.randn((4, feature_dim), device=fabric.device)
            y_baseline = baseline_model(x)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(fabric.device)
            baseline_start_memory = torch.cuda.max_memory_allocated(fabric.device)
            fabric.backward(y_baseline.mean().abs())
            del y_baseline
            for optimizer in baseline_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # FSDP sometimes holds onto grad memory until the next forward
                # pass. In order to provide a fair comparison (and thus an
                # accurate check that moving the step call into backward actually
                # delivers the expected memory savings) we need to "help" the
                # baseline case a bit.
                param_handles = _get_fsdp_handles(baseline_model._forward_module)
                for h in param_handles:
                    h._clear_grads_if_needed()

            baseline_peak_memory = torch.cuda.max_memory_allocated(fabric.device)

        # `.step()` interleaved with `.backward()`
        with status_checker.guard_region(f"({step + 1} / {num_steps}) Optimizer in backward"):
            y_test = test_model(x)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(fabric.device)
            test_start_memory = torch.cuda.memory_allocated(fabric.device)
            with fsdp_overlap_step_with_backward(test_optimizers, test_model):
                fabric.backward(y_test.mean().abs())
                del y_test

            test_peak_memory = torch.cuda.max_memory_allocated(fabric.device)

        # Make sure the parameter updates match.
        with status_checker.guard_region(f"({step + 1} / {num_steps}) Check equality"):
            for idx, (p0, p1) in enumerate(zip(baseline_model.parameters(), test_model.parameters())):
                assert (p0 == p1).all(), (step, idx, p0, p1)

        # The first step is going to be odd due to lazy initialization of optimizer state.
        if not step:
            continue

        with status_checker.guard_region(f"({step + 1} / {num_steps}) Confirm memory reduction"):
            baseline_delta = baseline_peak_memory - baseline_start_memory
            test_delta = test_peak_memory - test_start_memory
            assert (baseline_delta - test_delta) >= lower_savings_bound, (baseline_delta, test_delta)
            assert (baseline_delta - test_delta) <= upper_savings_bound, (baseline_delta, test_delta)

    status_checker.finalize()
    assert (pid := os.getpid()) == status_checker.pids[0], f"Orphan worker: {pid}"
