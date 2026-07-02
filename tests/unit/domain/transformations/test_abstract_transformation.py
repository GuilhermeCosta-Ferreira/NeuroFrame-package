# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

from neuroframe.domain.transformations.abstract_transformation import (
    AbstractTransformation,
)



# ================================================================
# 1. Section: Contract Tests
# ================================================================
@pytest.mark.unit
def test_abstract_transformation_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AbstractTransformation()  # type: ignore
