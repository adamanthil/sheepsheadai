"""Frozen evaluation panels shared by orchestrators and instruments.

Membership lives in ``models/manifest.json`` (``panels``): artifact keys in
frozen order, each pinned by sha256 and fetched into ``models/`` by
``sheepshead.analysis.fetch_models``. Do not edit a panel's membership
without an explicit pre-registration update — every recorded panel number
depends on its exact anchor set; a new field is a new panel name.

PANEL-A (pre-registered 2026-07; see notebooks/Architecture_Ablation_202607.md)
is the yardstick every pre-registered bar reads. PANEL-B (2026-09-16;
Training_Program_Redesign §5.3) holds the strongest artifacts across two
architectures and two training regimes — the production 30M, the
perceiver-shared-v2 lineage's release, its iteration-11 checkpoint and its
8M league seed — so it keeps discriminating at strong skill; recorded only.

Members are identified by key: ``panel_files`` returns repo-root-relative
``models/<key>.pt`` paths, and the field's seat order and design hash are
taken over those paths. Reads recorded before the 2026-10 move named
``runs/...`` paths, so they have a different design hash and seat draws:
same members and deals, comparable statistically but not pairable per deal.
Run panel commands from the repository root.
"""

from pathlib import Path

from sheepshead.analysis.fetch_models import load_manifest


def panel_files(name: str) -> list[str]:
    """Repo-root-relative paths of panel ``name``'s members, in frozen order."""
    manifest = load_manifest()
    return [
        str(Path("models") / manifest["artifacts"][key]["asset"])
        for key in manifest["panels"][name]
    ]
