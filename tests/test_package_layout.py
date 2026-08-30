"""Package layout after the stages / pipelines split."""

from __future__ import annotations

import importlib

import pytest

from st123.pipelines import Pipeline
from st123.stages import Stage


def test_old_top_level_stage_packages_are_gone():
    for name in (
        'st123.alignment',
        'st123.mast',
        'st123.mosaic',
        'st123.photometry',
        'st123.frame',
        'st123.stages.frame',
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


@pytest.mark.parametrize(
    'module_name',
    [
        'st123.stages',
        'st123.stages.stage',
        'st123.stages.alignment',
        'st123.stages.download',
        'st123.stages.mosaic',
        'st123.stages.photometry',
        'st123.datamodels',
        'st123.datamodels.hst',
        'st123.datamodels.jwst',
        'st123.pipelines',
        'st123.pipelines.pipeline',
    ],
)
def test_stage_and_pipeline_packages_import(module_name: str):
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_stage_primitive_merges_arguments():
    class ToyStage(Stage):
        name = 'toy'
        ARGUMENTS = {'toy_flag': True}

        def _perform(self):
            return 'ok'

    assert 'base_dir' in ToyStage.ARGUMENTS
    assert ToyStage.ARGUMENTS['toy_flag'] is True
    stage = ToyStage(ncores=4)
    assert stage.ncores == 4
    assert stage.toy_flag is True


def test_pipeline_apply_is_todo():
    with pytest.raises(NotImplementedError, match='not implemented yet'):
        Pipeline().apply()
