import importlib.util
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_github_pages_redirects.py"


def load_redirect_generator_module():
    spec = importlib.util.spec_from_file_location("generate_github_pages_redirects", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_sample_site(source_dir: Path, *, include_images: bool = True) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)

    if include_images:
        (source_dir / "images").mkdir(parents=True)
        (source_dir / "images" / "logo.png").write_bytes(b"png")


class RedirectSiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)
        self.source_dir = self.tmp_path / "site"
        self.output_dir = self.tmp_path / "gh-pages-redirect"
        self.redirect_generator = load_redirect_generator_module()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_generate_github_pages_redirects(self) -> None:
        write_sample_site(self.source_dir)

        redirect_count = self.redirect_generator.build_redirect_site(
            source_dir=self.source_dir,
            output_dir=self.output_dir,
            base_url=self.redirect_generator.DEFAULT_BASE_URL,
            pages_base_path=self.redirect_generator.DEFAULT_PAGES_BASE_PATH,
        )

        root_redirect = (self.output_dir / "index.html").read_text(encoding="utf-8")
        quickstart_redirect = (self.output_dir / "quickstart" / "index.html").read_text(encoding="utf-8")
        legacy_slug_redirect = (self.output_dir / "example-llamafiles" / "index.html").read_text(encoding="utf-8")
        github_redirect = (self.output_dir / "AGENTS" / "index.html").read_text(encoding="utf-8")
        not_found_redirect = (self.output_dir / "404.html").read_text(encoding="utf-8")

        self.assertGreater(redirect_count, 0)
        self.assertIn("https://docs.mozilla.ai/llamafile/", root_redirect)
        self.assertIn("https://docs.mozilla.ai/llamafile/getting-started/quickstart/", quickstart_redirect)
        self.assertIn(
            "https://docs.mozilla.ai/llamafile/getting-started/pre-built-llamafiles/",
            legacy_slug_redirect,
        )
        self.assertIn("https://github.com/mozilla-ai/llamafile/blob/main/docs/AGENTS.md", github_redirect)
        self.assertTrue((self.output_dir / "quickstart.html").exists())
        self.assertTrue((self.output_dir / "agents" / "index.html").exists())
        self.assertTrue((self.output_dir / ".nojekyll").exists())
        self.assertEqual((self.output_dir / "images" / "logo.png").read_bytes(), b"png")
        self.assertIn('const pagesBasePath = "/llamafile";', not_found_redirect)
        self.assertIn('"quickstart": "https://docs.mozilla.ai/llamafile/getting-started/quickstart/"', not_found_redirect)

    def test_serialize_json_for_script_escapes_html_sensitive_characters(self) -> None:
        serialized = self.redirect_generator.serialize_json_for_script(
            {
                "target": "https://docs.mozilla.ai/llamafile/</script>?q=<unsafe>&x=1",
            }
        )

        self.assertIn("\\u003c/script\\u003e", serialized)
        self.assertIn("\\u003cunsafe\\u003e", serialized)
        self.assertIn("\\u0026x=1", serialized)
        self.assertNotIn("</script>", serialized)

    def test_build_redirect_site_replaces_existing_output_and_skips_missing_images(self) -> None:
        write_sample_site(self.source_dir, include_images=False)
        self.output_dir.mkdir(parents=True)
        (self.output_dir / "stale.txt").write_text("old\n", encoding="utf-8")

        redirect_count = self.redirect_generator.build_redirect_site(
            source_dir=self.source_dir,
            output_dir=self.output_dir,
            base_url="https://docs.mozilla.ai/llamafile",
            pages_base_path="llamafile",
        )

        self.assertGreater(redirect_count, 0)
        self.assertFalse((self.output_dir / "stale.txt").exists())
        self.assertTrue((self.output_dir / "index.html").exists())
        self.assertFalse((self.output_dir / "images").exists())

    def test_parse_args_and_main_success(self) -> None:
        custom_output_dir = self.tmp_path / "custom-output"
        write_sample_site(self.source_dir, include_images=False)

        argv = [
            "generate_github_pages_redirects.py",
            "--source-dir",
            str(self.source_dir),
            "--output-dir",
            str(custom_output_dir),
            "--base-url",
            "https://docs.mozilla.ai/llamafile",
            "--pages-base-path",
            "llamafile",
        ]

        with mock.patch("sys.argv", argv):
            args = self.redirect_generator.parse_args()

        self.assertEqual(args.source_dir, self.source_dir)
        self.assertEqual(args.output_dir, custom_output_dir)
        self.assertEqual(args.base_url, "https://docs.mozilla.ai/llamafile")
        self.assertEqual(args.pages_base_path, "llamafile")

        with mock.patch("sys.argv", argv):
            with mock.patch("sys.stdout", new_callable=lambda: __import__("io").StringIO()) as stdout:
                exit_code = self.redirect_generator.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Done - ", stdout.getvalue())
        self.assertTrue((custom_output_dir / "404.html").exists())

    def test_main_returns_error_for_missing_source(self) -> None:
        missing_source = self.tmp_path / "missing-site"

        argv = [
            "generate_github_pages_redirects.py",
            "--source-dir",
            str(missing_source),
            "--output-dir",
            str(self.output_dir),
        ]

        with mock.patch("sys.argv", argv):
            with mock.patch("sys.stderr", new_callable=lambda: __import__("io").StringIO()) as stderr:
                exit_code = self.redirect_generator.main()

        self.assertEqual(exit_code, 1)
        self.assertIn(f"Input directory does not exist: {missing_source}", stderr.getvalue())

    def test_script_entrypoint_exits_with_main_status(self) -> None:
        entrypoint_output_dir = self.tmp_path / "entrypoint-output"
        write_sample_site(self.source_dir, include_images=False)

        argv = [
            str(SCRIPT_PATH),
            "--source-dir",
            str(self.source_dir),
            "--output-dir",
            str(entrypoint_output_dir),
        ]

        with mock.patch("sys.argv", argv):
            with self.assertRaises(SystemExit) as exc_info:
                runpy.run_path(str(SCRIPT_PATH), run_name="__main__")

        self.assertEqual(exc_info.exception.code, 0)
        self.assertTrue((entrypoint_output_dir / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
