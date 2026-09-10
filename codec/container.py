"""Version-1 container and lossless FASTQ side-stream framing."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import operator
import os
import shutil
import struct
import tempfile
import zlib
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from .fastq_stream import MAX_BATCH_READS


CONTAINER_FORMAT = "fastq-direct-quality-container"
CONTAINER_VERSION = 1
CONTAINER_MAGIC = b"FQDC0001"
CONTAINER_FLAGS = 0
SECTION_NAMES = ("header_gzip", "base_gzip", "plus_gzip", "quality_range")
MAX_METADATA_BYTES = 16 << 20
MAX_FIELD_BYTES = 1 << 32

# magic, version, flags, JSON bytes, CRC32(JSON)
_PREFIX = struct.Struct("<8sHHII")
CONTAINER_PREFIX_BYTES = _PREFIX.size

_SIDE_MAGICS = {
    "header": b"FQH1",
    "base": b"FQB1",
    "plus": b"FQP1",
}
_FIELD_PREFIX = struct.Struct("<QB")
_PLUS_PREFIX = struct.Struct("<QBB")
_ENDING_TO_CODE = {b"": 0, b"\n": 1, b"\r\n": 2}
_CODE_TO_ENDING = {value: key for key, value in _ENDING_TO_CODE.items()}


class ContainerError(ValueError):
    """Base class for container and side-stream validation errors."""


class TruncatedContainerError(ContainerError):
    """Raised when the physical file is shorter than its declared structure."""


class ContainerIntegrityError(ContainerError):
    """Raised when metadata, section, gzip, or reconstructed data checks fail."""


@dataclass(frozen=True)
class SectionSource:
    name: str
    path: Path
    uncompressed_length: int


@dataclass(frozen=True)
class SectionInfo:
    name: str
    offset: int
    length: int
    uncompressed_length: int
    crc32: int


@dataclass(frozen=True)
class ContainerInfo:
    path: Path
    metadata: Dict[str, Any]
    sections: Tuple[SectionInfo, ...]
    header_bytes: int
    file_size: int

    def section(self, name: str) -> SectionInfo:
        for section in self.sections:
            if section.name == name:
                return section
        raise KeyError(name)

    @contextmanager
    def open_section(self, name: str) -> Iterator[BinaryIO]:
        section = self.section(name)
        raw = _BoundedSectionReader(
            self.path, self.header_bytes + section.offset, section.length
        )
        buffered = io.BufferedReader(raw)
        try:
            yield buffered
        finally:
            buffered.close()

    def read_section(self, name: str) -> bytes:
        with self.open_section(name) as handle:
            data = handle.read()
        section = self.section(name)
        if len(data) != section.length:
            raise TruncatedContainerError(f"section {name!r} is truncated")
        return data


@dataclass(frozen=True)
class SideRecord:
    field: bytes
    line_ending: bytes
    quality_line_ending: Optional[bytes] = None


class _BoundedSectionReader(io.RawIOBase):
    def __init__(self, path: Path, offset: int, length: int) -> None:
        super().__init__()
        self._handle = path.open("rb")
        self._handle.seek(offset)
        self._remaining = length

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        if self._remaining == 0:
            return 0
        requested = min(len(buffer), self._remaining)
        data = self._handle.read(requested)
        if not data:
            raise TruncatedContainerError("container section ended unexpectedly")
        buffer[: len(data)] = data
        self._remaining -= len(data)
        return len(data)

    def close(self) -> None:
        if not self.closed:
            self._handle.close()
        super().close()


def _crc32_file(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_metadata(metadata: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContainerError("container metadata is not canonical JSON data") from exc


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ContainerError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ContainerError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ContainerError(f"{name} must be at least {minimum}")
    return result


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContainerError(f"{name} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContainerError(f"{name} is not hexadecimal") from exc
    return value.lower()


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("format") != CONTAINER_FORMAT:
        raise ContainerError("unsupported container format")
    if _integer(metadata.get("format_version"), name="format_version") != CONTAINER_VERSION:
        raise ContainerError("unsupported container format version")
    batch_reads = _integer(metadata.get("batch_reads"), name="batch_reads", minimum=1)
    if batch_reads > MAX_BATCH_READS:
        raise ContainerError(f"batch_reads must not exceed {MAX_BATCH_READS}")
    read_count = _integer(metadata.get("read_count"), name="read_count")
    expected_batches = (read_count + batch_reads - 1) // batch_reads
    if _integer(metadata.get("batch_count"), name="batch_count") != expected_batches:
        raise ContainerError("batch_count does not match read_count and batch_reads")
    expected_last_batch = (
        0 if read_count == 0 else ((read_count - 1) % batch_reads) + 1
    )
    if (
        _integer(
            metadata.get("last_batch_read_count"), name="last_batch_read_count"
        )
        != expected_last_batch
    ):
        raise ContainerError("last_batch_read_count does not match read_count")
    _integer(metadata.get("quality_symbol_count"), name="quality_symbol_count")
    if not isinstance(metadata.get("source_basename"), str):
        raise ContainerError("source_basename must be a string")
    _integer(
        metadata.get("source_uncompressed_size"), name="source_uncompressed_size"
    )
    _integer(metadata.get("source_compressed_size"), name="source_compressed_size")
    _validate_sha256(
        metadata.get("source_uncompressed_sha256"), name="source_uncompressed_sha256"
    )

    checkpoint = metadata.get("checkpoint")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_config"), dict
    ):
        raise ContainerError("checkpoint metadata or model_config is missing")
    _validate_sha256(checkpoint.get("sha256"), name="checkpoint.sha256")
    _integer(
        checkpoint.get("checkpoint_schema_version"),
        name="checkpoint_schema_version",
        minimum=1,
    )
    if not isinstance(checkpoint.get("feature_schema"), dict):
        raise ContainerError("checkpoint feature_schema is missing")

    gzip_streams = metadata.get("gzip_side_streams")
    if not isinstance(gzip_streams, dict):
        raise ContainerError("gzip_side_streams metadata is missing")
    compresslevel = _integer(
        gzip_streams.get("compresslevel"), name="gzip compresslevel"
    )
    if compresslevel > 9:
        raise ContainerError("gzip compresslevel must be in [0, 9]")
    _integer(
        gzip_streams.get("record_schema_version"),
        name="side record schema version",
        minimum=1,
    )
    if gzip_streams.get("quality_line_ending_location") != "plus_gzip":
        raise ContainerError("quality line ending location must be plus_gzip")

    quantization = metadata.get("probability_quantization")
    if not isinstance(quantization, dict):
        raise ContainerError("probability_quantization metadata is missing")
    _integer(quantization.get("version"), name="quantization version", minimum=1)
    _integer(quantization.get("total"), name="quantization total", minimum=1)
    _integer(
        quantization.get("quality_alphabet_size"),
        name="quality_alphabet_size",
        minimum=1,
    )
    _integer(quantization.get("phred_offset"), name="phred_offset")

    range_coder = metadata.get("range_coder")
    if not isinstance(range_coder, dict):
        raise ContainerError("range_coder metadata is missing")
    _integer(range_coder.get("version"), name="range coder version", minimum=1)
    if _integer(range_coder.get("stream_count"), name="stream_count", minimum=1) != 1:
        raise ContainerError("version 1 requires exactly one quality range stream")
    _integer(
        range_coder.get("payload_bit_count"), name="range payload bit count"
    )

    runtime = metadata.get("inference_runtime")
    if not isinstance(runtime, dict):
        raise ContainerError("inference_runtime metadata is missing")
    if not all(
        isinstance(runtime.get(name), str)
        for name in ("torch_version", "device_type", "dtype")
    ):
        raise ContainerError("inference_runtime fields must be strings")


def _section_infos(sources: Sequence[SectionSource]) -> Tuple[SectionInfo, ...]:
    if tuple(source.name for source in sources) != SECTION_NAMES:
        raise ContainerError(f"sections must appear exactly as {SECTION_NAMES}")
    offset = 0
    sections = []
    for source in sources:
        path = Path(source.path)
        if not path.is_file():
            raise FileNotFoundError(path)
        length = path.stat().st_size
        uncompressed = _integer(
            source.uncompressed_length,
            name=f"{source.name}.uncompressed_length",
        )
        sections.append(
            SectionInfo(
                name=source.name,
                offset=offset,
                length=length,
                uncompressed_length=uncompressed,
                crc32=_crc32_file(path),
            )
        )
        offset += length
    return tuple(sections)


def write_container(
    path: Path,
    metadata: Mapping[str, Any],
    section_sources: Sequence[SectionSource],
) -> ContainerInfo:
    """Atomically write a validated container from four section files."""

    output_path = Path(path)
    if output_path.exists():
        raise FileExistsError(f"{output_path}: output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "sections" in metadata:
        raise ContainerError("caller metadata must not predefine sections")

    sections = _section_infos(section_sources)
    complete_metadata = dict(metadata)
    complete_metadata["sections"] = [asdict(section) for section in sections]
    _validate_metadata(complete_metadata)
    metadata_bytes = _canonical_metadata(complete_metadata)
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise ContainerError("container metadata exceeds the version-1 size limit")
    prefix = _PREFIX.pack(
        CONTAINER_MAGIC,
        CONTAINER_VERSION,
        CONTAINER_FLAGS,
        len(metadata_bytes),
        zlib.crc32(metadata_bytes) & 0xFFFFFFFF,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(prefix)
            output.write(metadata_bytes)
            for source in section_sources:
                with Path(source.path).open("rb") as section_input:
                    shutil.copyfileobj(section_input, output, length=1 << 20)
            output.flush()
            os.fsync(output.fileno())
        os.replace(str(temporary_path), str(output_path))
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return ContainerInfo(
        path=output_path,
        metadata=complete_metadata,
        sections=sections,
        header_bytes=len(prefix) + len(metadata_bytes),
        file_size=output_path.stat().st_size,
    )


def _parse_section_infos(raw_sections: Any) -> Tuple[SectionInfo, ...]:
    if not isinstance(raw_sections, list) or len(raw_sections) != len(SECTION_NAMES):
        raise ContainerError("container must contain exactly four sections")
    sections = []
    expected_offset = 0
    for expected_name, raw in zip(SECTION_NAMES, raw_sections):
        if not isinstance(raw, dict) or raw.get("name") != expected_name:
            raise ContainerError(f"expected section {expected_name!r}")
        offset = _integer(raw.get("offset"), name=f"{expected_name}.offset")
        length = _integer(raw.get("length"), name=f"{expected_name}.length")
        uncompressed = _integer(
            raw.get("uncompressed_length"),
            name=f"{expected_name}.uncompressed_length",
        )
        checksum = _integer(raw.get("crc32"), name=f"{expected_name}.crc32")
        if checksum > 0xFFFFFFFF:
            raise ContainerError(f"{expected_name}.crc32 exceeds uint32")
        if offset != expected_offset:
            raise ContainerError("container sections must be contiguous and ordered")
        sections.append(
            SectionInfo(
                name=expected_name,
                offset=offset,
                length=length,
                uncompressed_length=uncompressed,
                crc32=checksum,
            )
        )
        expected_offset += length
    return tuple(sections)


def read_container(path: Path, *, verify_checksums: bool = True) -> ContainerInfo:
    """Parse the full structure and optionally stream-verify every section CRC32."""

    container_path = Path(path)
    with container_path.open("rb") as handle:
        prefix = handle.read(_PREFIX.size)
        if len(prefix) != _PREFIX.size:
            raise TruncatedContainerError("container is truncated before its prefix")
        magic, version, flags, metadata_length, metadata_crc32 = _PREFIX.unpack(prefix)
        if magic != CONTAINER_MAGIC or version != CONTAINER_VERSION:
            raise ContainerError("invalid container magic/version")
        if flags != CONTAINER_FLAGS:
            raise ContainerError("unsupported container flags")
        if metadata_length > MAX_METADATA_BYTES:
            raise ContainerError("container metadata length exceeds the version-1 limit")
        metadata_bytes = handle.read(metadata_length)
        if len(metadata_bytes) != metadata_length:
            raise TruncatedContainerError("container metadata is truncated")
    if (zlib.crc32(metadata_bytes) & 0xFFFFFFFF) != metadata_crc32:
        raise ContainerIntegrityError("container metadata checksum mismatch")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContainerError("container metadata is not valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict):
        raise ContainerError("container metadata must be a JSON object")
    if _canonical_metadata(metadata) != metadata_bytes:
        raise ContainerError("container metadata is not in canonical JSON form")
    _validate_metadata(metadata)
    sections = _parse_section_infos(metadata.get("sections"))
    header_bytes = _PREFIX.size + metadata_length
    expected_size = header_bytes + sum(section.length for section in sections)
    observed_size = container_path.stat().st_size
    if observed_size < expected_size:
        raise TruncatedContainerError(
            f"container advertises {expected_size} bytes but has {observed_size}"
        )
    if observed_size > expected_size:
        raise ContainerError("container has trailing bytes")

    info = ContainerInfo(
        path=container_path,
        metadata=metadata,
        sections=sections,
        header_bytes=header_bytes,
        file_size=observed_size,
    )
    if verify_checksums:
        for section in sections:
            checksum = 0
            with info.open_section(section.name) as handle:
                while True:
                    chunk = handle.read(1 << 20)
                    if not chunk:
                        break
                    checksum = zlib.crc32(chunk, checksum)
            if (checksum & 0xFFFFFFFF) != section.crc32:
                raise ContainerIntegrityError(
                    f"container section {section.name!r} checksum mismatch"
                )
    return info


class SideStreamWriter:
    """Write one record-aligned gzip side stream and count raw framing bytes."""

    def __init__(self, path: Path, kind: str, *, compresslevel: int = 6) -> None:
        if kind not in _SIDE_MAGICS:
            raise ValueError(f"unknown side stream kind {kind!r}")
        if compresslevel < 0 or compresslevel > 9:
            raise ValueError("gzip compresslevel must be in [0, 9]")
        self.path = Path(path)
        self.kind = kind
        self.compresslevel = compresslevel
        self.uncompressed_length = 0
        self._raw: Optional[BinaryIO] = None
        self._gzip: Optional[gzip.GzipFile] = None

    def __enter__(self) -> "SideStreamWriter":
        self._raw = self.path.open("wb")
        self._gzip = gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=self.compresslevel,
            fileobj=self._raw,
            mtime=0,
        )
        self._write(_SIDE_MAGICS[self.kind])
        return self

    def _write(self, data: bytes) -> None:
        if self._gzip is None:
            raise RuntimeError("side stream writer is not open")
        self._gzip.write(data)
        self.uncompressed_length += len(data)

    def write_record(
        self,
        field: bytes,
        line_ending: bytes,
        *,
        quality_line_ending: Optional[bytes] = None,
    ) -> None:
        try:
            ending_code = _ENDING_TO_CODE[line_ending]
        except KeyError as exc:
            raise ContainerError("unsupported FASTQ line ending") from exc
        if self.kind == "plus":
            if quality_line_ending is None:
                raise ContainerError("plus records must carry the quality line ending")
            try:
                quality_code = _ENDING_TO_CODE[quality_line_ending]
            except KeyError as exc:
                raise ContainerError("unsupported quality line ending") from exc
            prefix = _PLUS_PREFIX.pack(len(field), ending_code, quality_code)
        else:
            if quality_line_ending is not None:
                raise ContainerError("only plus records carry quality line endings")
            prefix = _FIELD_PREFIX.pack(len(field), ending_code)
        self._write(prefix)
        self._write(field)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        try:
            if self._gzip is not None:
                self._gzip.close()
        finally:
            if self._raw is not None:
                self._raw.close()
            self._gzip = None
            self._raw = None


class SideStreamReader:
    """Read and validate one decompressed side stream record at a time."""

    def __init__(
        self,
        compressed: BinaryIO,
        kind: str,
        *,
        expected_uncompressed_length: int,
    ) -> None:
        if kind not in _SIDE_MAGICS:
            raise ValueError(f"unknown side stream kind {kind!r}")
        self.kind = kind
        self.expected_uncompressed_length = expected_uncompressed_length
        self.uncompressed_length = 0
        self._gzip = gzip.GzipFile(fileobj=compressed, mode="rb")
        magic = self._read_exact(4, "side-stream magic")
        if magic != _SIDE_MAGICS[kind]:
            raise ContainerIntegrityError(f"invalid {kind} side-stream magic/version")

    def _read_exact(self, length: int, description: str) -> bytes:
        chunks = []
        remaining = length
        while remaining:
            chunk = self._gzip.read(remaining)
            if not chunk:
                raise TruncatedContainerError(f"truncated {self.kind} {description}")
            chunks.append(chunk)
            self.uncompressed_length += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read_record(self) -> SideRecord:
        prefix_size = _PLUS_PREFIX.size if self.kind == "plus" else _FIELD_PREFIX.size
        prefix = self._read_exact(prefix_size, "record prefix")
        if self.kind == "plus":
            field_length, ending_code, quality_code = _PLUS_PREFIX.unpack(prefix)
        else:
            field_length, ending_code = _FIELD_PREFIX.unpack(prefix)
            quality_code = None
        if field_length > MAX_FIELD_BYTES:
            raise ContainerError(f"{self.kind} field exceeds version-1 size limit")
        try:
            line_ending = _CODE_TO_ENDING[ending_code]
            quality_ending = (
                _CODE_TO_ENDING[quality_code] if quality_code is not None else None
            )
        except KeyError as exc:
            raise ContainerIntegrityError(
                f"invalid line-ending code in {self.kind} side stream"
            ) from exc
        field = self._read_exact(field_length, "record field")
        return SideRecord(field, line_ending, quality_ending)

    def finish(self) -> None:
        extra = self._gzip.read(1)
        self.uncompressed_length += len(extra)
        if extra:
            raise ContainerIntegrityError(f"{self.kind} side stream has extra records")
        if self.uncompressed_length != self.expected_uncompressed_length:
            raise ContainerIntegrityError(
                f"{self.kind} side stream uncompressed length mismatch"
            )

    def close(self) -> None:
        self._gzip.close()


__all__ = [
    "CONTAINER_FORMAT",
    "CONTAINER_MAGIC",
    "CONTAINER_PREFIX_BYTES",
    "CONTAINER_VERSION",
    "ContainerError",
    "ContainerInfo",
    "ContainerIntegrityError",
    "SECTION_NAMES",
    "SectionInfo",
    "SectionSource",
    "SideRecord",
    "SideStreamReader",
    "SideStreamWriter",
    "TruncatedContainerError",
    "read_container",
    "sha256_file",
    "write_container",
]
