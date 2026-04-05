from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class RegriddingParameters:
    ramp_up_time: float | None = None
    ramp_down_time: float | None = None
    flat_top_time: float | None = None
    acquisition_delay_time: float | None = None


@dataclass(frozen=True)
class HeaderInfo:
    encoded_matrix: tuple[int, int, int]
    recon_matrix: tuple[int, int, int]
    encoding_limits: dict[str, int]
    regridding: RegriddingParameters


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _find_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if _local_name(child.tag) == name:
            return child
    return None


def _find_path(element: ET.Element, names: list[str]) -> ET.Element | None:
    current = element
    for name in names:
        current = _find_child(current, name)
        if current is None:
            return None
    return current


def _find_text(element: ET.Element, path: list[str], default: int = 0) -> int:
    node = _find_path(element, path)
    if node is None or node.text is None:
        return default
    return int(node.text)


def _find_any_text(element: ET.Element, name: str) -> float | None:
    for child in element.iter():
        if _local_name(child.tag) == name and child.text is not None:
            try:
                return float(child.text)
            except ValueError:
                return None
    return None


def parse_ismrmrd_header(header: bytes | str) -> HeaderInfo:
    if isinstance(header, bytes):
        header = header.decode()

    root = ET.fromstring(header)
    encoding = next((node for node in root.iter() if _local_name(node.tag) == "encoding"), None)
    if encoding is None:
        raise ValueError("ISMRMRD header does not contain an encoding section")

    encoded_space = _find_path(encoding, ["encodedSpace", "matrixSize"])
    recon_space = _find_path(encoding, ["reconSpace", "matrixSize"])
    if encoded_space is None or recon_space is None:
        raise ValueError("ISMRMRD header is missing matrix size information")

    encoded_matrix = (
        _find_text(encoded_space, ["x"]),
        _find_text(encoded_space, ["y"]),
        _find_text(encoded_space, ["z"], default=1),
    )
    recon_matrix = (
        _find_text(recon_space, ["x"]),
        _find_text(recon_space, ["y"]),
        _find_text(recon_space, ["z"], default=1),
    )

    encoding_limits = {
        "kspace_encoding_step_1_center": _find_text(
            encoding,
            ["encodingLimits", "kspace_encoding_step_1", "center"],
            default=encoded_matrix[1] // 2,
        ),
        "kspace_encoding_step_1_max": _find_text(
            encoding,
            ["encodingLimits", "kspace_encoding_step_1", "maximum"],
            default=max(encoded_matrix[1] - 1, 0),
        ),
    }

    regridding = RegriddingParameters(
        ramp_up_time=_find_any_text(root, "rampUpTime"),
        ramp_down_time=_find_any_text(root, "rampDownTime"),
        flat_top_time=_find_any_text(root, "flatTopTime"),
        acquisition_delay_time=_find_any_text(root, "acqDelayTime"),
    )
    return HeaderInfo(
        encoded_matrix=encoded_matrix,
        recon_matrix=recon_matrix,
        encoding_limits=encoding_limits,
        regridding=regridding,
    )
