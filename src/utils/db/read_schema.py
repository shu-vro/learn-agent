"""Build Pydantic read models from SQLAlchemy column mappings (single source of truth)."""

from __future__ import annotations

from typing import Any, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, create_model


def read_schema_for_orm_columns(
    orm_class: Any, *, name: str | None = None
) -> Type[BaseModel]:
    """Return a ``BaseModel`` subclass with one field per mapped column.

    Field names and nullability come from the ORM mapper, so new table columns
    on ``orm_class`` are reflected here on the next import (no hand-maintained
    ``ProjectOut``). Only ``column_attrs`` are included, not relationships.
    """
    mapper = orm_class.__mapper__
    model_name = name or f"{orm_class.__name__}Read"
    field_defs: dict[str, tuple[Any, Any]] = {}

    for col_attr in mapper.column_attrs:
        key = col_attr.key
        column = col_attr.columns[0]
        try:
            py_t: Any = column.type.python_type
        except NotImplementedError:
            py_t = Any

        if column.nullable:
            field_defs[key] = (Optional[py_t], Field(default=None))
        else:
            field_defs[key] = (py_t, ...)

    return create_model(
        model_name,
        __config__=ConfigDict(from_attributes=True, extra="ignore"),
        **field_defs,
    )
