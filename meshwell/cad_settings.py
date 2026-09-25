"""Single source of truth for meshwell's CAD-stage settings.

The CAD stage (:func:`meshwell.cad`) and the mesh stage (:func:`meshwell.mesh`)
run independently -- often in different processes, with the ``.xao`` file as
the only hand-off. Historically each stage restated its own defaults
(``point_tolerance=1e-3``, ``min_arc_points=5``, ...) and the mesh stage
guessed CAD-stage values with ``or 1e-3`` fallbacks. :class:`CADSettings`
replaces those guesses:

* the ``DEFAULT_*`` constants below are the only place package-level defaults
  are spelled out;
* :class:`CADSettings` resolves derived values (fuzzy ladder) once and
  validates them;
* the CAD stage embeds the settings in the ``.xao`` it writes
  (:meth:`CADSettings.to_xao_element` / :meth:`CADSettings.append_to_xao`),
  and the mesh stage recovers them (:meth:`CADSettings.from_xao`,
  :meth:`CADSettings.resolve_for_intake`), raising on missing metadata or on a
  mismatch with caller-supplied values.

XAO placement: gmsh's XAO reader rejects anything placed before
``<geometry>`` (extra root attributes included) but ignores extra children
after ``</geometry>``. The metadata block is therefore always written as the
LAST child of ``<XAO>``::

    <XAO ...>
      <geometry>...</geometry>
      <groups>...</groups>
      <meshwell version="1" meshwell_version="...">
        <cad_settings format="json">{"point_tolerance": 0.001, ...}</cad_settings>
      </meshwell>
    </XAO>

This module intentionally has no heavy imports (no gmsh / OCP) so every other
meshwell module can import the defaults without cycles.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Package-level defaults. Every other module should reference these rather
# than restating the literal values.
# ---------------------------------------------------------------------------
DEFAULT_POINT_TOLERANCE: float = 1e-3
DEFAULT_PERTURBATION: float = 0.0
DEFAULT_IDENTIFY_ARCS: bool = False
DEFAULT_MIN_ARC_POINTS: int = 5
DEFAULT_ARC_TOLERANCE: float = 1e-3

XAO_METADATA_TAG = "meshwell"
XAO_SETTINGS_TAG = "cad_settings"
XAO_METADATA_SCHEMA_VERSION = "1"

# The metadata block is the last child of <XAO>, so it always lives in the
# file's tail. Reading only the tail avoids parsing the (potentially huge)
# inlined BREP CDATA.
_XAO_TAIL_BYTES = 1 << 20
_XAO_BLOCK_RE = re.compile(
    rb"<" + XAO_METADATA_TAG.encode() + rb"\b.*?</" + XAO_METADATA_TAG.encode() + rb">",
    re.S,
)


class CADSettingsError(ValueError):
    """Base class for CAD-settings provenance errors."""


class MissingCADSettingsError(CADSettingsError):
    """The intake ``.xao`` carries no meshwell metadata and none was supplied."""


class CADSettingsMismatchError(CADSettingsError):
    """Two sources of CAD settings (file / caller / model) disagree."""


@dataclass(frozen=True)
class CADSettings:
    """Frozen, validated bundle of CAD-stage numerical settings.

    Attributes:
        point_tolerance: Coordinate quantization / grid-snap tolerance.
        perturbation: Analytic outward offset for same-``mesh_order``
            boundaries. ``0.0`` is canonical-exact mode.
        identify_arcs: Scene-wide arc identification flag.
        min_arc_points: Minimum run length for arc fitting.
        arc_tolerance: Circle-fit tolerance.
        cut_fuzzy_value: ``BRepAlgoAPI_Cut`` fuzzy. ``None`` resolves to
            ``0.8 * perturbation`` if ``perturbation > 0`` else
            ``0.5 * fragment_fuzzy_value``.
        fragment_fuzzy_value: ``BOPAlgo_Builder`` fragment fuzzy. ``None``
            resolves to ``point_tolerance``.

    ``None`` fuzzy inputs are resolved in ``__post_init__``, so two settings
    that describe the same effective CAD run compare equal regardless of
    whether the derived values were spelled out.
    """

    point_tolerance: float = DEFAULT_POINT_TOLERANCE
    perturbation: float = DEFAULT_PERTURBATION
    identify_arcs: bool = DEFAULT_IDENTIFY_ARCS
    min_arc_points: int = DEFAULT_MIN_ARC_POINTS
    arc_tolerance: float = DEFAULT_ARC_TOLERANCE
    cut_fuzzy_value: float | None = None
    fragment_fuzzy_value: float | None = None

    def __post_init__(self) -> None:
        """Type-normalize, resolve derived fuzzy values and validate."""
        _set = object.__setattr__

        for name in ("point_tolerance", "perturbation", "arc_tolerance"):
            value = getattr(self, name)
            if value is None or isinstance(value, bool):
                raise TypeError(f"CADSettings.{name} must be a float, got {value!r}")
            _set(self, name, float(value))
        if not isinstance(self.identify_arcs, bool):
            raise TypeError(
                f"CADSettings.identify_arcs must be a bool, got {self.identify_arcs!r}"
            )
        if (
            isinstance(self.min_arc_points, bool)
            or int(self.min_arc_points) != self.min_arc_points
        ):
            raise TypeError(
                f"CADSettings.min_arc_points must be an int, got {self.min_arc_points!r}"
            )
        _set(self, "min_arc_points", int(self.min_arc_points))

        if not (math.isfinite(self.point_tolerance) and self.point_tolerance > 0):
            raise ValueError(
                f"CADSettings.point_tolerance must be finite and > 0, got {self.point_tolerance}"
            )
        if not (math.isfinite(self.perturbation) and self.perturbation >= 0):
            raise ValueError(
                f"CADSettings.perturbation must be finite and >= 0, got {self.perturbation}"
            )
        if not (math.isfinite(self.arc_tolerance) and self.arc_tolerance > 0):
            raise ValueError(
                f"CADSettings.arc_tolerance must be finite and > 0, got {self.arc_tolerance}"
            )
        if self.min_arc_points < 3:
            raise ValueError(
                f"CADSettings.min_arc_points must be >= 3, got {self.min_arc_points}"
            )

        cut, fragment = self.resolve_fuzzy(
            point_tolerance=self.point_tolerance,
            perturbation=self.perturbation,
            cut_fuzzy_value=self.cut_fuzzy_value,
            fragment_fuzzy_value=self.fragment_fuzzy_value,
        )
        _set(self, "cut_fuzzy_value", cut)
        _set(self, "fragment_fuzzy_value", fragment)

        from meshwell.validation import validate_tolerance_ladder

        validate_tolerance_ladder(
            perturbation=self.perturbation,
            cut_fuzzy_value=self.cut_fuzzy_value,
            fragment_fuzzy_value=self.fragment_fuzzy_value,
        )

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_fuzzy(
        *,
        point_tolerance: float,
        perturbation: float,
        cut_fuzzy_value: float | None,
        fragment_fuzzy_value: float | None,
    ) -> tuple[float, float]:
        """Return ``(cut_fuzzy_value, fragment_fuzzy_value)`` with defaults applied.

        The only place the fuzzy-ladder defaults are defined; ``CAD_OCC``
        delegates here. See ``CAD_OCC.__init__`` for the rationale of each
        regime.
        """
        fragment = (
            float(point_tolerance)
            if fragment_fuzzy_value is None
            else float(fragment_fuzzy_value)
        )
        if cut_fuzzy_value is not None:
            cut = float(cut_fuzzy_value)
        elif perturbation > 0:
            cut = 0.8 * perturbation
        else:
            cut = 0.5 * fragment
        return cut, fragment

    @property
    def resolve_snap(self) -> float:
        """Snap distance for ``prepare_entities`` / InterfaceTag resolution."""
        return max(self.perturbation, self.point_tolerance)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_kwargs(
        cls, cad_settings: CADSettings | None = None, /, **overrides: Any
    ) -> CADSettings:
        """Build settings from either a ``CADSettings`` or loose kwargs.

        ``None``-valued overrides mean "not specified". Supplying both a
        ``cad_settings`` object and any non-``None`` override is ambiguous
        and raises ``TypeError``.
        """
        specified = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(specified) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise TypeError(f"Unknown CADSettings field(s): {sorted(unknown)}")
        if cad_settings is not None:
            if not isinstance(cad_settings, CADSettings):
                raise TypeError(
                    f"cad_settings must be a CADSettings, got {type(cad_settings).__name__}"
                )
            if specified:
                raise TypeError(
                    "Pass either cad_settings=CADSettings(...) or individual "
                    f"settings kwargs, not both (got {sorted(specified)})."
                )
            return cad_settings
        return cls(**specified)

    def replace(self, **changes: Any) -> CADSettings:
        """Return a copy with ``changes`` applied (fuzzy values re-derived if unset)."""
        data = self.to_dict()
        # Re-derive fuzzy values unless explicitly carried over or changed.
        if "point_tolerance" in changes or "perturbation" in changes:
            data.pop("cut_fuzzy_value")
            data.pop("fragment_fuzzy_value")
        data.update(changes)
        return type(self).from_dict(data)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-compatible dict (resolved values)."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CADSettings:
        """Deserialize from :meth:`to_dict` output. Unknown keys raise."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise CADSettingsError(f"Unknown CADSettings field(s): {sorted(unknown)}")
        return cls(**data)

    def to_json(self) -> str:
        """Canonical JSON encoding (sorted keys; floats round-trip exactly)."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> CADSettings:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def to_xao_element(self) -> ET.Element:
        """Return the ``<meshwell>`` metadata element (append as LAST child of ``<XAO>``)."""
        root = ET.Element(
            XAO_METADATA_TAG,
            version=XAO_METADATA_SCHEMA_VERSION,
            meshwell_version=_meshwell_version(),
        )
        settings_el = ET.SubElement(root, XAO_SETTINGS_TAG, format="json")
        settings_el.text = self.to_json()
        return root

    def append_to_xao(self, xao_path: Path | str) -> None:
        """Embed these settings into an existing ``.xao`` file (in place).

        Used for XAO files written by gmsh itself (``gmsh.write``), where
        :func:`meshwell.occ_xao_writer.write_xao` isn't in the loop. Any
        existing meshwell block is replaced, so the call is idempotent.
        """
        path = Path(xao_path)
        text = path.read_text(encoding="utf-8")
        text = re.sub(
            r"\s*<" + XAO_METADATA_TAG + r"\b.*?</" + XAO_METADATA_TAG + r">",
            "",
            text,
            flags=re.S,
        )
        close = text.rfind("</XAO>")
        if close < 0:
            raise CADSettingsError(f"{path} is not an XAO file (no closing </XAO>).")
        element = self.to_xao_element()
        ET.indent(element, space="  ", level=1)
        block = "  " + ET.tostring(element, encoding="unicode") + "\n"
        head = text[:close]
        if not head.endswith("\n"):
            head += "\n"
        path.write_text(head + block + text[close:], encoding="utf-8")

    @classmethod
    def from_xao(cls, xao_path: Path | str) -> CADSettings | None:
        """Read embedded settings from a ``.xao``; ``None`` if absent.

        Raises:
            CADSettingsError: if a meshwell block exists but is malformed or
                uses an unsupported schema version.
        """
        path = Path(xao_path).with_suffix(".xao")
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - _XAO_TAIL_BYTES))
            tail = fh.read()
        matches = list(_XAO_BLOCK_RE.finditer(tail))
        if not matches:
            return None
        try:
            # Safe: only the regex-bounded <meshwell> snippet is parsed; it
            # cannot carry a DTD, so entity-expansion attacks do not apply.
            element = ET.fromstring(matches[-1].group(0))  # noqa: S314
        except ET.ParseError as exc:
            raise CADSettingsError(
                f"Malformed meshwell metadata in {path}: {exc}"
            ) from exc
        version = element.get("version")
        if version != XAO_METADATA_SCHEMA_VERSION:
            raise CADSettingsError(
                f"Unsupported meshwell XAO metadata version {version!r} in {path} "
                f"(expected {XAO_METADATA_SCHEMA_VERSION!r})."
            )
        settings_el = element.find(XAO_SETTINGS_TAG)
        if settings_el is None or settings_el.get("format") != "json":
            raise CADSettingsError(
                f"meshwell metadata in {path} has no JSON <{XAO_SETTINGS_TAG}> block."
            )
        try:
            return cls.from_json(settings_el.text or "")
        except (json.JSONDecodeError, TypeError) as exc:
            raise CADSettingsError(
                f"Malformed <{XAO_SETTINGS_TAG}> payload in {path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Consistency checks (mesh-stage intake)
    # ------------------------------------------------------------------
    def diff(self, other: CADSettings) -> dict[str, tuple[Any, Any]]:
        """Return ``{field: (self_value, other_value)}`` for differing fields."""
        return {
            f.name: (getattr(self, f.name), getattr(other, f.name))
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != getattr(other, f.name)
        }

    def check_matches(
        self,
        other: CADSettings | None,
        *,
        self_label: str,
        other_label: str,
    ) -> None:
        """Raise :class:`CADSettingsMismatchError` if ``other`` differs."""
        if other is None:
            return
        if not isinstance(other, CADSettings):
            raise TypeError(
                f"{other_label} must be a CADSettings, got {type(other).__name__}"
            )
        delta = self.diff(other)
        if delta:
            lines = ", ".join(
                f"{k}: {self_label}={a!r} vs {other_label}={b!r}"
                for k, (a, b) in sorted(delta.items())
            )
            raise CADSettingsMismatchError(f"CAD settings mismatch -- {lines}")

    def check_point_tolerance(
        self, point_tolerance: float | None, *, label: str
    ) -> None:
        """Raise if an explicitly supplied ``point_tolerance`` disagrees."""
        if point_tolerance is None:
            return
        if float(point_tolerance) != self.point_tolerance:
            raise CADSettingsMismatchError(
                f"CAD settings mismatch -- point_tolerance: CAD stage used "
                f"{self.point_tolerance!r} but {label}={point_tolerance!r}"
            )

    @classmethod
    def resolve_for_intake(
        cls,
        xao_path: Path | str,
        supplied: CADSettings | None = None,
        *,
        point_tolerance: float | None = None,
    ) -> CADSettings:
        """Resolve the CAD settings a mesh-stage function must use for ``xao_path``.

        Rules:

        * file has metadata, nothing supplied -> file settings;
        * file has metadata and ``supplied`` given -> must be equal, else
          :class:`CADSettingsMismatchError`;
        * file has no metadata -> ``supplied`` must be a ``CADSettings``, else
          :class:`MissingCADSettingsError`;
        * an explicit ``point_tolerance`` must equal the resolved one.
        """
        if supplied is not None and not isinstance(supplied, CADSettings):
            raise TypeError(
                f"cad_settings must be a CADSettings, got {type(supplied).__name__}"
            )
        stored = cls.from_xao(xao_path)
        if stored is None and supplied is None:
            raise MissingCADSettingsError(
                f"{xao_path} carries no meshwell CAD-settings metadata (it was not "
                "written by meshwell.cad / write_xao(cad_settings=...)). Pass "
                "cad_settings=CADSettings(...) describing how it was generated."
            )
        if stored is not None:
            stored.check_matches(supplied, self_label="xao", other_label="cad_settings")
            settings = stored
        else:
            settings = supplied
        settings.check_point_tolerance(
            point_tolerance, label="point_tolerance argument"
        )
        return settings


def _meshwell_version() -> str:
    try:
        from importlib.metadata import version

        return version("meshwell")
    except Exception:
        return "unknown"
