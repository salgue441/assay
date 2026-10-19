"""
Build one repository epoch into symbol inventories and documentation units.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass

from assay.corpus.discovery import DEFAULT_EXCLUDED_DIRS, DEFAULT_MAX_FILE_BYTES, discover_sources
from assay.corpus.layout import LanguageLayout, get_layout
from assay.corpus.manifest import RepoSpec
from assay.corpus.snapshot import Snapshot, SnapshotFetcher
from assay.corpus.units import DEFAULT_POLICY, Unit, UnitPolicy, select_units
from assay.extract import SourceLanguage, SymbolExtractor, SymbolInventory, get_extractor


@dataclass(frozen=True, slots=True)
class SnapshotCorpus:
    snapshot: Snapshot
    inventories: tuple[SymbolInventory, ...]
    units: tuple[Unit, ...]


class CorpusBuilder:
    """
    Fetch a snapshot, extract every source file and select its units.

    Collaborators are injected; the defaults resolve layouts and extractors
    from their registries.
    """

    def __init__(
        self,
        fetcher: SnapshotFetcher,
        *,
        policy: UnitPolicy = DEFAULT_POLICY,
        excluded_dirs: Collection[str] = DEFAULT_EXCLUDED_DIRS,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        layouts: Callable[[SourceLanguage], LanguageLayout] = get_layout,
        extractors: Callable[[SourceLanguage], SymbolExtractor] = get_extractor,
    ) -> None:
        self._fetcher = fetcher
        self._policy = policy
        self._excluded_dirs = excluded_dirs
        self._max_file_bytes = max_file_bytes
        self._layouts = layouts
        self._extractors = extractors

    def build(self, repo: RepoSpec, epoch: int) -> SnapshotCorpus:
        checkout = self._fetcher.fetch(repo, epoch)
        snapshot = checkout.snapshot
        layout = self._layouts(repo.language)
        extractor = self._extractors(repo.language)

        inventories: list[SymbolInventory] = []
        units: list[Unit] = []
        for source in discover_sources(
            checkout.root,
            layout,
            excluded_dirs=self._excluded_dirs,
            max_file_bytes=self._max_file_bytes,
        ):
            inventory = extractor.extract(source.text, module=source.module, path=str(source.path))
            inventories.append(inventory)
            units.extend(
                select_units(
                    inventory,
                    source,
                    repo=snapshot.repo,
                    commit=snapshot.commit,
                    policy=self._policy,
                )
            )

        return SnapshotCorpus(snapshot=snapshot, inventories=tuple(inventories), units=tuple(units))
