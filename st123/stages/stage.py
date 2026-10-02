"""
Shared stage primitive for st123.

Library code for each CLI stage lives under this package (alignment, mosaic,
download, photometry). Command-line wrappers remain in
:mod:`st123.scripts` until they are folded into these packages. Pipelines
that sequence stages live in :mod:`st123.pipelines`.

Modeled on HISPEC DRP ``BasePrimitive``: each concrete stage declares
``ARGUMENTS`` (name -> default), implements :meth:`_perform`, and is invoked
through :meth:`apply`. Stages do not parse argv; the options API does.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ['Stage']


class Stage:
    """Base class for a st123 processing stage.

    Parameters
    ----------
    **kwargs
        Values for keys in :attr:`ARGUMENTS`. Unknown names are logged and
        ignored.
    """

    name: str = 'stage'
    ARGUMENTS: dict[str, Any] = {
        'base_dir': None,
        'ncores': 1,
        'verbose': False,
        'dry_run': False,
        'existing_box': None,
        'stamp_id': 'sn',
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        merged: dict[str, Any] = {}
        for base in reversed(cls.mro()):
            own = base.__dict__.get('ARGUMENTS')
            if isinstance(own, dict):
                merged.update(own)
        cls.ARGUMENTS = merged

    def __init__(self, **kwargs: Any) -> None:
        self._output: Any = None
        self._status: str | None = None
        self.set_args(**kwargs)

    def set_args(self, **kwargs: Any) -> None:
        allowed = set(self.ARGUMENTS)
        for key, default in self.ARGUMENTS.items():
            setattr(self, key, kwargs.get(key, default))
        for key in kwargs:
            if key not in allowed:
                logger.error(
                    'Invalid argument %r for %s. Valid: %s',
                    key,
                    self.__class__.__name__,
                    sorted(allowed),
                )

    def args_namespace(self) -> argparse.Namespace:
        """Build an argparse-like namespace from current argument values."""
        return argparse.Namespace(
            **{key: getattr(self, key, self.ARGUMENTS[key]) for key in self.ARGUMENTS}
        )

    def _pre_condition(self) -> bool:
        return True

    def _perform(self) -> Any:
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement _perform()'
        )

    def apply(self, **kwargs: Any) -> Any:
        """Run the stage: logging, pre-condition, :meth:`_perform`."""
        from st123.scripts.utils.options import configure_logging_from_args
        from st123.utils.logging import shutdown_logging

        self.set_args(**kwargs)
        configure_logging_from_args(self.args_namespace(), self.name)
        try:
            if not self._pre_condition():
                logger.info('Skipping %s', self.__class__.__name__)
                self._status = 'SKIPPED'
                return None
            logger.info('Applying %s', self.__class__.__name__)
            self._output = self._perform()
            self._status = 'COMPLETE'
            return self._output
        except Exception:
            self._status = 'FAILED'
            logger.exception('Failed executing stage %s', self.__class__.__name__)
            raise
        finally:
            shutdown_logging()
