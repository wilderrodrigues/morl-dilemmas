# MIT License - 2026
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from uu.ai.thesis.core.functions import RandomNumberGenerator


def test_generate_creates_player_streams_and_seed_file(tmp_path: Path) -> None:
    """Generate streams and persist the child seed metadata."""
    master_seed = 42

    rng = RandomNumberGenerator(master_seed)
    rng.generate(tmp_path)

    seed_file = tmp_path / "child_seeds.txt"

    assert seed_file.exists()
    assert len(rng.player_streams) == 2
    assert all(len(streams) == 5 for streams in rng.player_streams)

    seed_contents = seed_file.read_text()
    assert seed_contents.count("SeedSequence") >= 2

    regenerated_rng = RandomNumberGenerator(master_seed)
    regenerated_rng.generate(tmp_path)

    first_run_values = [
        [stream.integers(0, 1000) for stream in player_streams]
        for player_streams in rng.player_streams
    ]
    second_run_values = [
        [stream.integers(0, 1000) for stream in player_streams]
        for player_streams in regenerated_rng.player_streams
    ]

    assert first_run_values == second_run_values
