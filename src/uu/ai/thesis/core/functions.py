# MIT License - 2026
"""Utility helpers for deterministic random-number stream generation.

This module currently provides a small wrapper around NumPy seed sequences to
create reproducible player-specific random-number generators for experiments.
"""
from pathlib import Path

from numpy.random import SeedSequence, default_rng

from uu import logger


class RandomNumberGenerator:
    """Create deterministic random number streams for each player.

    Parameters
    ----------
    master_seed : int
        Root seed used to derive reproducible player-specific generators.
    n_players : int, optional
        Number of players for which independent random-number stream groups are
        created.
    n_children : int, optional
        Number of child generators spawned for each player.

    Attributes
    ----------
    player_streams : list[list]
        Nested list of NumPy random-number generators grouped by player.
    """

    def __init__(self, master_seed: int | SeedSequence, n_players: int = 2, n_children: int = 5) -> None:
        """Initialize the random stream generator.

        Parameters
        ----------
        master_seed : int
            Root seed used to derive reproducible child seed sequences.
        n_players : int
            Number of players to generate.
        n_children : int
            Number of children to generate.
        """
        self.master_seed = master_seed
        self.n_players = n_players
        self.n_children = n_children
        self.player_streams = [list() for _ in range(self.n_players)]

    def generate(self, destination_folder: Path, seed_path: Path | None = None) -> None:
        """Generate player-specific random streams and persist child seeds.

        One child seed is created per player from the master seed. Each child
        seed is then expanded into five random number generators, which are
        stored in ``player_streams``. The child seed metadata is written to a
        ``child_seeds.txt`` file inside the destination folder.

        Parameters
        ----------
        destination_folder : Path
            Directory where the child seed metadata will be stored.
        seed_path: Path | None, optional
            Child seed metadata file name. Defaults to ``child_seeds.txt``.

        Raises
        ------
        ValueError
            If ``destination_folder`` is not a directory.
        """
        seed_seq = (
            self.master_seed
            if isinstance(self.master_seed, SeedSequence)
            else SeedSequence(self.master_seed)
        )

        child_seeds = seed_seq.spawn(self.n_players)

        if not destination_folder.is_dir():
            raise ValueError("Destination folder is not a directory.")

        if not destination_folder.exists():
            destination_folder.mkdir(parents=True, exist_ok=True)

        seed_file = seed_path or Path("child_seeds.txt")

        with open(destination_folder / seed_file, "w") as fp:
            for item in child_seeds:
                fp.write(f"{str(item)}\n")
            logger.info(f"Child seeds generated in {destination_folder} for players 1 and 2.")

        for player in range(self.n_players):
            grandchildren_player = child_seeds[player].spawn(self.n_children)
            self.player_streams[player] = [default_rng(s) for s in grandchildren_player]
