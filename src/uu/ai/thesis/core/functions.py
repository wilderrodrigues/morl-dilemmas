# MIT License - 2026
from pathlib import Path

from numpy.random import SeedSequence, default_rng
import os

from uu import logger


class RandomNumberGenerator:
    """Create deterministic random number streams for each player.

    Parameters
    ----------
    master_seed : int
        Root seed used to derive reproducible player-specific generators.
    """

    def __init__(self, master_seed: int) -> None:
        """Initialize the random stream generator.

        Parameters
        ----------
        master_seed : int
            Root seed used to derive reproducible child seed sequences.
        """
        self.master_seed = master_seed
        self.n_players = 2
        self.player_streams = [list() for _ in range(self.n_players)]

    def generate(self, destination_folder: Path) -> None:
        """Generate player-specific random streams and persist child seeds.

        One child seed is created per player from the master seed. Each child
        seed is then expanded into five random number generators, which are
        stored in ``player_streams``. The child seed metadata is written to a
        ``child_seeds.txt`` file inside the destination folder.

        Parameters
        ----------
        destination_folder : Path
            Directory where the child seed metadata will be stored.

        Raises
        ------
        ValueError
            If ``destination_folder`` is not a directory.
        """
        seed_seq = SeedSequence(self.master_seed)

        child_seeds = seed_seq.spawn(self.n_players)

        if not destination_folder.is_dir():
            raise ValueError('destination folder is not a directory.')

        if not destination_folder.exists():
            destination_folder.mkdir(parents=True, exist_ok=True)

        with open(f"{os.fspath(destination_folder)}/child_seeds.txt", "w") as fp:
            for item in child_seeds:
                fp.write(f"{str(item)}\n")
            logger.info(f"child seeds generated in {destination_folder} for players 1 and 2.")

        for player in range(self.n_players):
            grandchildren_player = child_seeds[player].spawn(5)
            self.player_streams[player] = [default_rng(s) for s in grandchildren_player]
