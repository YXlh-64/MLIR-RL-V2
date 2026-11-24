"""Read MLIR files with large weight literals elided (support for M1–M3 CLIs on full models).

Full models carry gigabytes of inline weights. Their values never matter to the
fusion layer: classification only reads structure, and extracted units take
weights as arguments. Large hex literals and resource blobs are replaced by zero
splats of the right type (via v6 `full_model.materialize_elided_resources`).
"""

import re

from rl_autoschedular_v6.full_model import materialize_elided_resources

_HEX_LITERAL = re.compile(r'dense<"0x[0-9A-Fa-f]{64,}">')
_RESOURCE_REF = re.compile(r'dense_resource<[\w.$-]+>')


def strip_weights(line: str) -> str:
    """Replace large hex literals and named resource references by the elided marker."""
    line = _HEX_LITERAL.sub('dense_resource<__elided__>', line)
    return _RESOURCE_REF.sub('dense_resource<__elided__>', line)


def read_mlir(path: str) -> str:
    """Read `path` line by line, eliding weights and dropping the `{-# ... #-}` resource section."""
    lines: list[str] = []
    in_resources = False
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.lstrip().startswith('{-#'):
                in_resources = True
            if not in_resources:
                lines.append(strip_weights(line))
            if in_resources and '#-}' in line:
                in_resources = False
    return materialize_elided_resources(''.join(lines))
