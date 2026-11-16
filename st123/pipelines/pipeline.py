"""
Campaign pipelines that sequence st123 stages.

Concrete pipelines are not implemented yet. :class:`Pipeline` is the wrapper
all future campaign classes will subclass (HISPEC ``BasePipeline`` analog).
Until then, run stages from :mod:`st123.scripts`
(``download``, ``align``, ``mosaic``, ``dolphot-prep``, ...).
"""

from __future__ import annotations

import logging
from typing import Any

from st123.stages.stage import Stage

logger = logging.getLogger(__name__)

__all__ = ['Pipeline']


class Pipeline:
    """Sequence of :class:`~st123.stages.stage.Stage` objects.

    ``STAGES`` maps alias -> stage class. :meth:`apply` will run them in
    insertion order once campaign pipelines are implemented.
    """

    name: str = 'pipeline'
    ARGUMENTS: dict[str, Any] = {
        'base_dir': None,
        'ncores': 1,
        'verbose': False,
        'dry_run': False,
    }
    STAGES: dict[str, type[Stage]] = {}

    def __init__(self, **kwargs: Any) -> None:
        self._output: Any = None
        self._status: str | None = None
        for key, default in self.ARGUMENTS.items():
            setattr(self, key, kwargs.get(key, default))

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Run the stage sequence.

        Raises
        ------
        NotImplementedError
            Always, until a concrete campaign pipeline is added.
        """
        raise NotImplementedError(
            'st123 pipelines are not implemented yet. Run individual stages '
            'from the command line (download, align, mosaic, dolphot-prep, '
            'run-dolphot, ...) or call library APIs under st123.stages.'
        )
