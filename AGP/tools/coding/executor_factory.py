#!/usr/bin/env python
# -*- coding: utf-8 -*-


from AGP.utils.log import logger
from AGP.environment.tools.coding.python_executor import PyExecutor
from AGP.environment.tools.coding.executor_types import Executor

EXECUTOR_MAPPING = {
    "py": PyExecutor,
    "python": PyExecutor,
}

def executor_factory(lang: str) -> Executor:

    if lang not in EXECUTOR_MAPPING:
        raise ValueError(f"Invalid language for executor: {lang}")

    executor_class = EXECUTOR_MAPPING[lang]
    return executor_class()