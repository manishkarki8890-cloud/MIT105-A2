"""Minimal pure-Python reader for Apple Numbers (.numbers) tables.
Decodes IWA chunks (Snappy-compressed protobuf) without external packages."""
import struct, zipfile, sys, datetime
from collections import defaultdict


def snappy_decompress(data):
    pos, shift, length = 0, 0, 0
    while True:
        b = data[pos]; pos += 1
        length |= (b & 0x7F) << shift
        if b < 0x80: break
        shift += 7
    out = bytearray()
    n = len(data)
    while pos < n:
        tag = data[pos]; pos += 1
        t = tag & 3
        if t == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(data[pos:pos + nb], 'little'); pos += nb
            ln += 1
            out += data[pos:pos + ln]; pos += ln
            continue
        if t == 1:
            ln = ((tag >> 2) & 7) + 4
            off = ((tag >> 5) << 8) | data[pos]; pos += 1
        elif t == 2:
            ln = (tag >> 2) + 1
            off = int.from_bytes(data[pos:pos + 2], 'little'); pos += 2
        else:
            ln = (tag >> 2) + 1
            off = int.from_bytes(data[pos:pos + 4], 'little'); pos += 4
        start = len(out) - off
        for i in range(ln):
            out.append(out[start + i])
    return bytes(out)


def read_varint(buf, pos):
    shift = res = 0
    while True:
        b = buf[pos]; pos += 1
        res |= (b & 0x7F) << shift
        if b < 0x80: return res, pos
        shift += 7


def decode_pb(buf):
    """Return dict field -> list of raw values (int or bytes)."""
    pos, out = 0, defaultdict(list)
    n = len(buf)
    while pos < n:
        key, pos = read_varint(buf, pos)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = read_varint(buf, pos)
        elif wt == 1:
            v = buf[pos:pos + 8]; pos += 8
        elif wt == 2:
            ln, pos = read_varint(buf, pos)
            v = buf[pos:pos + ln]; pos += ln
        elif wt == 5:
            v = buf[pos:pos + 4]; pos += 4
        else:
            raise ValueError('bad wire type %d' % wt)
        out[f].append(v)
    return out


def iwa_objects(raw):
    data = bytearray()
    pos = 0
    while pos < len(raw):
        ln = int.from_bytes(raw[pos + 1:pos + 4], 'little')
        data += snappy_decompress(raw[pos + 4:pos + 4 + ln])
        pos += 4 + ln
    data = bytes(data)
    pos, objs = 0, []
    while pos < len(data):
        ln, pos = read_varint(data, pos)
        info = decode_pb(data[pos:pos + ln]); pos += ln
        ident = info[1][0]
        for mi in info[2]:
            m = decode_pb(mi)
            mlen = m[3][0]
            objs.append((ident, m[1][0], data[pos:pos + mlen]))
            pos += mlen
    return objs


def unpack_decimal128(b):
    exp = (((b[15] & 0x7F) << 7) | (b[14] >> 1)) - 0x1820
    mant = b[14] & 1
    for i in range(13, -1, -1):
        mant = mant * 256 + b[i]
    if b[15] & 0x80: mant = -mant
    return float(mant) * (10.0 ** exp)


EPOCH = datetime.datetime(2001, 1, 1)


def parse_cell(buf, strings):
    if len(buf) < 12: return None
    ctype = buf[1]
    flags = struct.unpack('<I', buf[8:12])[0]
    off = 12
    d128 = dbl = secs = sid = None
    if flags & 0x1: d128 = unpack_decimal128(buf[off:off + 16]); off += 16
    if flags & 0x2: dbl = struct.unpack('<d', buf[off:off + 8])[0]; off += 8
    if flags & 0x4: secs = struct.unpack('<d', buf[off:off + 8])[0]; off += 8
    if flags & 0x8: sid = struct.unpack('<i', buf[off:off + 4])[0]; off += 4
    if ctype == 0: return None
    if ctype in (2, 10, 7):
        return d128 if d128 is not None else dbl
    if ctype == 3 or ctype == 9:
        return strings.get(sid, '')
    if ctype == 5:
        return (EPOCH + datetime.timedelta(seconds=secs)).isoformat() if secs is not None else None
    if ctype == 6:
        return bool(dbl if dbl is not None else d128)
    return d128 if d128 is not None else dbl


def read_numbers(path):
    z = zipfile.ZipFile(path)
    allobjs = {}
    for name in z.namelist():
        if name.endswith('.iwa'):
            for ident, typ, payload in iwa_objects(z.read(name)):
                allobjs[ident] = (typ, payload, name)
    # string tables: TableDataList type 6005, listType (1) == 1 STRING
    stringlists = {}
    for ident, (typ, p, name) in allobjs.items():
        if typ == 6005:
            m = decode_pb(p)
            if m[1] and m[1][0] == 1:
                d = {}
                for e in m[3]:
                    em = decode_pb(e)
                    if em[3]: d[em[1][0]] = em[3][0].decode('utf-8')
                stringlists[ident] = d
    tables = []
    for ident, (typ, p, name) in allobjs.items():
        if typ != 6001: continue
        tm = decode_pb(p)
        nrows, ncols = tm[6][0], tm[7][0]
        ds = decode_pb(tm[4][0])
        sref = decode_pb(ds[4][0])[1][0]
        strings = stringlists.get(sref, {})
        ts = decode_pb(ds[3][0])
        grid = [[None] * ncols for _ in range(nrows)]
        for t in ts[1]:
            tt = decode_pb(t)
            tileid = tt[1][0]
            tref = decode_pb(tt[2][0])[1][0]
            tile = decode_pb(allobjs[tref][1])
            for ri in tile[5]:
                r = decode_pb(ri)
                row = tileid * 256 + r[1][0]
                if row >= nrows: continue
                wide = bool(r[8][0]) if r[8] else False
                storage = r[6][0] if r[6] else b''
                offs = r[7][0] if r[7] else b''
                if not storage: continue
                offvals = struct.unpack('<%dH' % (len(offs) // 2), offs)
                valid = [(c, o * 4 if wide else o) for c, o in enumerate(offvals) if o != 0xFFFF]
                for i, (c, o) in enumerate(valid):
                    end = valid[i + 1][1] if i + 1 < len(valid) else len(storage)
                    if c < ncols:
                        grid[row][c] = parse_cell(storage[o:end], strings)
        tname = tm[8][0].decode() if tm[8] else 'table'
        tables.append((tname, grid))
    return tables


if __name__ == '__main__':
    import csv
    tables = read_numbers(sys.argv[1])
    for name, grid in tables:
        print(name, len(grid), len(grid[0]))
    name, grid = max(tables, key=lambda t: len(t[1]))
    with open(sys.argv[2], 'w', newline='') as f:
        csv.writer(f).writerows([['' if v is None else v for v in row] for row in grid])
