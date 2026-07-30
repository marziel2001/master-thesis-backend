from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    """Base class for every request and response schema.

    Pydantic reserves the ``model_`` field-name prefix and warns about fields
    like ``model_name`` / ``model_version``. Those names are part of the
    published API contract, so the guard is disabled instead of renaming them.
    """

    model_config = ConfigDict(protected_namespaces=())
