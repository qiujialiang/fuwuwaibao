from pathlib import Path
def get_attach_path():
    home_path=Path(__file__).parent.parent
    attach_path=home_path.joinpath('attach')
    if not attach_path.exists():
        attach_path.mkdir(parents=True)
    return attach_path