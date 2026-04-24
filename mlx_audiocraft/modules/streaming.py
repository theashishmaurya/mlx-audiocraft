# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Streaming module infrastructure.

from contextlib import contextmanager
import typing as tp

import mlx.core as mx
import mlx.nn as nn

State = tp.Dict[str, mx.array]


class StreamingModule(nn.Module):
    """Common API for streaming components (MLX version).

    Each streaming component has a streaming state, which is a dict[str, mx.array].
    By convention, the first dim of each tensor must be the batch size.

    If ``self._is_streaming`` is True, the component should use and remember
    the proper state inside ``self._streaming_state``.

    Usage::

        with module.streaming():
            ...

    This automatically resets the streaming state when exiting the context manager
    and propagates to all streaming children modules.
    """

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: State = {}
        self._is_streaming = False

    def _iter_streaming_modules(self) -> tp.Iterator[tp.Tuple[str, 'StreamingModule']]:
        """Iterate over all named StreamingModule children (including self)."""
        results: tp.List[tp.Tuple[str, 'StreamingModule']] = []

        def _collect(path: str, mod: nn.Module) -> nn.Module:
            if isinstance(mod, StreamingModule):
                results.append((path, mod))
            return mod

        self.apply_to_modules(_collect)
        return iter(results)

    def _apply_named_streaming(self, fn: tp.Callable[[str, 'StreamingModule'], None]):
        for name, module in self._iter_streaming_modules():
            fn(name, module)

    def _set_streaming(self, streaming: bool):
        def _set(name: str, module: StreamingModule):
            module._is_streaming = streaming
        self._apply_named_streaming(_set)

    @contextmanager
    def streaming(self):
        """Context manager to enter streaming mode. Reset streaming state on exit."""
        self._set_streaming(True)
        try:
            yield
        finally:
            self._set_streaming(False)
            self.reset_streaming()

    def reset_streaming(self):
        """Reset the streaming state."""
        def _reset(name: str, module: StreamingModule):
            module._streaming_state.clear()
        self._apply_named_streaming(_reset)

    def get_streaming_state(self) -> State:
        """Return the streaming state, including that of sub-modules."""
        state: State = {}

        def _add(name: str, module: StreamingModule):
            if name:
                name += "."
            for key, value in module._streaming_state.items():
                state[name + key] = value

        self._apply_named_streaming(_add)
        return state

    def set_streaming_state(self, state: State):
        """Set the streaming state, including that of sub-modules."""
        state = dict(state)

        def _set(name: str, module: StreamingModule):
            if name:
                name += "."
            module._streaming_state.clear()
            for key, value in list(state.items()):
                if key.startswith(name):
                    local_key = key[len(name):]
                    if '.' not in local_key:
                        module._streaming_state[local_key] = value
                        del state[key]

        self._apply_named_streaming(_set)
        assert len(state) == 0, list(state.keys())

    def flush(self, x: tp.Optional[mx.array] = None) -> tp.Optional[mx.array]:
        """Flush remaining outputs (e.g. final padding for convolutions)."""
        if x is None:
            return None
        else:
            return self(x)

    def named_modules(self, prefix: str = '') -> tp.Iterator[tp.Tuple[str, nn.Module]]:
        """Recursively yield (name, module) pairs, matching PyTorch's API."""
        yield prefix, self
        for attr_name in vars(self):
            child = getattr(self, attr_name, None)
            if isinstance(child, nn.Module):
                child_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
                if isinstance(child, StreamingModule):
                    yield from child.named_modules(prefix=child_prefix)
                else:
                    yield child_prefix, child
                    # Also recurse into non-streaming nn.Module children
                    for sub_name, sub_mod in _iter_nn_modules(child, child_prefix):
                        yield sub_name, sub_mod
            elif isinstance(child, (list, tuple)):
                for i, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        child_prefix = f"{prefix}.{attr_name}.{i}" if prefix else f"{attr_name}.{i}"
                        if isinstance(item, StreamingModule):
                            yield from item.named_modules(prefix=child_prefix)
                        else:
                            yield child_prefix, item
                            for sub_name, sub_mod in _iter_nn_modules(item, child_prefix):
                                yield sub_name, sub_mod


def _iter_nn_modules(module: nn.Module, prefix: str) -> tp.Iterator[tp.Tuple[str, nn.Module]]:
    """Recurse into a plain nn.Module to find nested modules."""
    for attr_name in vars(module):
        child = getattr(module, attr_name, None)
        if isinstance(child, nn.Module):
            child_prefix = f"{prefix}.{attr_name}"
            yield child_prefix, child
            yield from _iter_nn_modules(child, child_prefix)
        elif isinstance(child, (list, tuple)):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    child_prefix = f"{prefix}.{attr_name}.{i}"
                    yield child_prefix, item
                    yield from _iter_nn_modules(item, child_prefix)


class StreamingSequential(StreamingModule):
    """Streaming-compatible sequential container."""

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.layers = list(modules)

    def __call__(self, x: mx.array) -> mx.array:
        for module in self.layers:
            x = module(x)
        return x

    def __iter__(self):
        return iter(self.layers)

    def flush(self, x: tp.Optional[mx.array] = None) -> tp.Optional[mx.array]:
        for module in self.layers:
            if isinstance(module, StreamingModule):
                x = module.flush(x)
            elif x is not None:
                x = module(x)
        return x
