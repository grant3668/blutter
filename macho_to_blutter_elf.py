#!/usr/bin/env python3
"""Wrap a decrypted Flutter iOS App.framework/App Mach-O as a minimal ELF.

Blutter's loader only needs an ELF section table plus .dynsym/.dynstr entries
for the four Dart snapshot symbols. For Flutter iOS App.framework binaries
whose __TEXT vmaddr is 0, the Dart snapshot virtual addresses already match
file offsets. This script preserves the original snapshot layout and replaces
the first page with a small ELF header/section table that points at it.
"""

from __future__ import annotations

import argparse
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path


MH_MAGIC_64 = 0xFEEDFACF
LC_SEGMENT_64 = 0x19
LC_SYMTAB = 0x2

SNAPSHOT_SYMBOLS = [
    "_kDartVmSnapshotInstructions",
    "_kDartIsolateSnapshotInstructions",
    "_kDartVmSnapshotData",
    "_kDartIsolateSnapshotData",
]


@dataclass(frozen=True)
class Section:
    sectname: str
    segname: str
    addr: int
    size: int
    offset: int


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def read_cstr(data: bytes, offset: int) -> str:
    end = data.index(b"\0", offset)
    return data[offset:end].decode()


def parse_macho(path: Path) -> tuple[dict[str, int], list[Section]]:
    data = path.read_bytes()
    if len(data) < 32:
        raise ValueError("input is too small to be a Mach-O")

    magic, cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags, reserved = struct.unpack_from(
        "<IiiIIIII", data, 0
    )
    if magic != MH_MAGIC_64:
        raise ValueError(f"expected little-endian 64-bit Mach-O, got magic 0x{magic:x}")

    sections: list[Section] = []
    symoff = nsyms = stroff = strsize = None
    cursor = 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, cursor)
        if cmd == LC_SEGMENT_64:
            segname = data[cursor + 8 : cursor + 24].split(b"\0", 1)[0].decode()
            nsects = struct.unpack_from("<I", data, cursor + 64)[0]
            sec_cursor = cursor + 72
            for _ in range(nsects):
                raw = struct.unpack_from("<16s16sQQIIIIIIII", data, sec_cursor)
                sectname = raw[0].split(b"\0", 1)[0].decode()
                sec_segname = raw[1].split(b"\0", 1)[0].decode()
                sections.append(
                    Section(
                        sectname=sectname,
                        segname=sec_segname or segname,
                        addr=raw[2],
                        size=raw[3],
                        offset=raw[4],
                    )
                )
                sec_cursor += 80
        elif cmd == LC_SYMTAB:
            symoff, nsyms, stroff, strsize = struct.unpack_from("<IIII", data, cursor + 8)
        cursor += cmdsize

    if symoff is None or stroff is None or nsyms is None:
        raise ValueError("Mach-O has no LC_SYMTAB")

    symbols: dict[str, int] = {}
    for i in range(nsyms):
        n_strx, n_type, n_sect, n_desc, n_value = struct.unpack_from("<IBBHQ", data, symoff + i * 16)
        if n_strx == 0:
            continue
        name = read_cstr(data, stroff + n_strx)
        if name in SNAPSHOT_SYMBOLS:
            symbols[name] = n_value

    missing = [name for name in SNAPSHOT_SYMBOLS if name not in symbols]
    if missing:
        raise ValueError("missing Dart snapshot symbols: " + ", ".join(missing))

    return symbols, sections


def section_for_addr(sections: list[Section], addr: int) -> Section:
    for section in sections:
        if section.addr <= addr < section.addr + section.size:
            return section
    raise ValueError(f"no Mach-O section contains address 0x{addr:x}")


def build_elf_header(symbols: dict[str, int], sections: list[Section]) -> bytes:
    text = section_for_addr(sections, symbols["_kDartVmSnapshotInstructions"])
    const = section_for_addr(sections, symbols["_kDartVmSnapshotData"])
    if text.offset != text.addr or const.offset != const.addr:
        raise ValueError(
            "this wrapper requires Mach-O section file offsets to match vm addresses"
        )

    dynstr = b"\0"
    name_offsets: dict[str, int] = {}
    for name in SNAPSHOT_SYMBOLS:
        name_offsets[name] = len(dynstr)
        dynstr += name.encode() + b"\0"

    shstr_names = ["", ".dynstr", ".dynsym", ".text", ".rodata", ".shstrtab"]
    shstr = b""
    sh_name_offsets: dict[str, int] = {}
    for name in shstr_names:
        sh_name_offsets[name] = len(shstr)
        shstr += name.encode() + b"\0"

    dynstr_off = 0x100
    dynsym_off = align(dynstr_off + len(dynstr), 8)

    sorted_symbols = sorted(symbols.items(), key=lambda item: item[1])
    dynsym = bytearray(24)  # STN_UNDEF
    for name, value in sorted_symbols:
        owner = section_for_addr(sections, value)
        owner_idx = 3 if owner == text else 4 if owner == const else 0
        larger = [addr for _, addr in sorted_symbols if addr > value and section_for_addr(sections, addr) == owner]
        end = min(larger) if larger else owner.addr + owner.size
        size = end - value
        dynsym += struct.pack("<IBBHQQ", name_offsets[name], 0x11, 0, owner_idx, value, size)

    shstr_off = align(dynsym_off + len(dynsym), 8)
    shoff = align(shstr_off + len(shstr), 8)

    def sh(name: str, typ: int, flags: int, addr: int, off: int, size: int, link=0, info=0, addralign=1, entsize=0):
        return struct.pack(
            "<IIQQQQIIQQ",
            sh_name_offsets[name],
            typ,
            flags,
            addr,
            off,
            size,
            link,
            info,
            addralign,
            entsize,
        )

    section_headers = b"".join(
        [
            bytes(64),
            sh(".dynstr", 3, 0, dynstr_off, dynstr_off, len(dynstr), addralign=1),
            sh(".dynsym", 11, 0, dynsym_off, dynsym_off, len(dynsym), link=1, info=1, addralign=8, entsize=24),
            sh(".text", 1, 0x6, text.addr, text.offset, text.size, addralign=64),
            sh(".rodata", 1, 0x2, const.addr, const.offset, const.size, addralign=64),
            sh(".shstrtab", 3, 0, shstr_off, shstr_off, len(shstr), addralign=1),
        ]
    )

    if shoff + len(section_headers) > min(text.offset, const.offset):
        raise ValueError("ELF metadata does not fit before the first Dart snapshot section")

    ident = b"\x7fELF" + bytes([2, 1, 1, 0, 0]) + bytes(7)
    elf_header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        ident,
        3,      # ET_DYN
        183,    # EM_AARCH64
        1,
        0,
        0,
        shoff,
        0,
        64,
        56,
        0,
        64,
        6,
        5,
    )

    header = bytearray(shoff + len(section_headers))
    header[: len(elf_header)] = elf_header
    header[dynstr_off : dynstr_off + len(dynstr)] = dynstr
    header[dynsym_off : dynsym_off + len(dynsym)] = dynsym
    header[shstr_off : shstr_off + len(shstr)] = shstr
    header[shoff : shoff + len(section_headers)] = section_headers
    return bytes(header)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("macho", type=Path, help="decrypted App.framework/App Mach-O")
    parser.add_argument("elf", type=Path, help="output fake ELF path")
    args = parser.parse_args()

    symbols, sections = parse_macho(args.macho)
    header = build_elf_header(symbols, sections)

    args.elf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.macho, args.elf)
    with args.elf.open("r+b") as f:
        f.write(header)

    for name in SNAPSHOT_SYMBOLS:
        print(f"{name}: 0x{symbols[name]:x}")
    print(f"wrote {args.elf}")


if __name__ == "__main__":
    main()
