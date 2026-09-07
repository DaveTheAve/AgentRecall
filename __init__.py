"""Source-tree entry point for the native Hermes plugin.

The MemoryProvider implementation lives in :mod:`hermes_plugin`
so wheels and source checkouts exercise the same adapter code.
"""

try:
    from .hermes_plugin import *  # noqa: F403
    from .hermes_plugin import (
        __all__ as __all__,
    )
    from .hermes_plugin import (
        _default_config as _default_config,
    )
    from .hermes_plugin import (
        _normalize_config as _normalize_config,
    )
except ImportError:
    from hermes_plugin import *  # noqa: F403
    from hermes_plugin import (
        __all__ as __all__,
    )
    from hermes_plugin import (
        _default_config as _default_config,
    )
    from hermes_plugin import (
        _normalize_config as _normalize_config,
    )
