from pathlib import Path

def get_project_root() -> Path:
    """Get the root directory of the project."""
    script_dir = Path(__file__).parent
    if script_dir.name == "ssp_modeling":
        return script_dir.parent.parent
    return script_dir.parent.parent.parent


print(get_project_root())