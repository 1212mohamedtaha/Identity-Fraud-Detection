"""Registry of domain packs. Add yours here (see docs/guides/adding-a-pack.md)."""
from .cv import CVPack
from .identity import IdentityPack

PACKS = {
    IdentityPack.name: IdentityPack,
    CVPack.name: CVPack,
}


def get_pack(name, llm=None):
    if name not in PACKS:
        raise ValueError(f"Unknown pack {name!r}; choose from {sorted(PACKS)}")
    return PACKS[name](llm=llm)
