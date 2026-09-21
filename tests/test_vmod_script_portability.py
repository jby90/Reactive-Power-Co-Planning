from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vmod_orchestrators_do_not_hardcode_a_user_python_path():
    for path in ROOT.glob("run_vmod*.py"):
        text = path.read_text(encoding="utf-8")
        assert "C:/Users/" not in text
        assert "C:\\Users\\" not in text
        assert "sys.executable" in text
