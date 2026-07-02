"""HTTPS/TLS integration tests.

Covers the TLS support added via cpp-httplib's Mbed TLS backend:
serving over HTTPS (--ssl-cert-file/--ssl-key-file), downloading models
over HTTPS (-hf/--model-url), and rejection of untrusted certificates.

Toggle with the ``ssl`` marker: ``-m ssl`` / ``-m "not ssl"``.
``test_download_model_over_https`` needs network access (huggingface.co)
and is additionally marked ``online`` (skip with ``-m "not online"``);
the other tests run fully offline.
"""

import os
import platform
import shutil
import subprocess

import pytest
import requests

from utils.llamafile import LlamafileRunner

# Tiny model used for the network download test (~1 MiB)
HF_TEST_REPO = "ggml-org/models"
HF_TEST_FILE = "tinyllamas/stories260K.gguf"

OPENSSL_CNF = """\
[req]
distinguished_name = dn
x509_extensions = v3_req
prompt = no
[dn]
CN = localhost
[v3_req]
subjectAltName = DNS:localhost, IP:127.0.0.1
"""


@pytest.fixture(scope="module")
def ssl_cert(tmp_path_factory):
    """Generate a self-signed cert/key pair for localhost.

    Returns (cert_path, key_path). Skips if no openssl binary is available.
    The SAN config file keeps this portable across OpenSSL and LibreSSL
    (macOS ships LibreSSL, where -addext support varies).
    """
    if not shutil.which("openssl"):
        pytest.skip("openssl binary not available to generate a test certificate")

    tmp = tmp_path_factory.mktemp("ssl")
    cnf = tmp / "openssl.cnf"
    cnf.write_text(OPENSSL_CNF)
    cert, key = tmp / "cert.pem", tmp / "key.pem"

    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256",
            "-days", "2", "-nodes",
            "-keyout", str(key), "-out", str(cert),
            "-config", str(cnf),
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def wait_for_https_server(port, cert, timeout, poll_interval=1.0):
    """Poll https://127.0.0.1:port/health until 200, verifying against cert."""
    import time

    url = f"https://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=2, verify=str(cert)).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(poll_interval)
    return False


def base_exe_args(executable):
    """Mimic LlamafileRunner's sh-prefixed invocation for manual Popen."""
    if platform.system() != "Windows":
        return ["sh", os.path.abspath(executable)]
    return [os.path.abspath(executable)]


def fresh_cache_env(cache_dir):
    """Environment pointing the model download cache at a fresh directory."""
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache_dir)
    return env


@pytest.mark.ssl
@pytest.mark.server
class TestHttpsServing:
    """Serving over TLS with --ssl-cert-file/--ssl-key-file."""

    def test_server_serves_https(self, llamafile, server_port, timeouts, ssl_cert):
        """Server comes up over HTTPS and a verifying client can connect."""
        cert, key = ssl_cert
        proc = llamafile.start_server(
            port=server_port,
            extra_args=["--ssl-cert-file", str(cert), "--ssl-key-file", str(key)],
        )
        try:
            assert wait_for_https_server(
                server_port, cert, timeout=timeouts.server_ready
            ), "Server did not become ready over HTTPS"

            # Chain + hostname verification against our cert succeeds
            r = requests.get(
                f"https://127.0.0.1:{server_port}/props",
                timeout=timeouts.http_request,
                verify=str(cert),
            )
            assert r.status_code == 200
            assert "default_generation_settings" in r.text
        finally:
            proc.terminate()
            proc.wait()

    def test_https_server_rejects_plaintext(
        self, llamafile, server_port, timeouts, ssl_cert
    ):
        """A plain-HTTP request to the TLS port must not get an HTTP 200."""
        cert, key = ssl_cert
        proc = llamafile.start_server(
            port=server_port,
            extra_args=["--ssl-cert-file", str(cert), "--ssl-key-file", str(key)],
        )
        try:
            assert wait_for_https_server(
                server_port, cert, timeout=timeouts.server_ready
            ), "Server did not become ready over HTTPS"

            with pytest.raises(requests.RequestException):
                requests.get(f"http://127.0.0.1:{server_port}/health", timeout=5)
        finally:
            proc.terminate()
            proc.wait()


@pytest.mark.ssl
class TestHttpsDownload:
    """Model downloads over HTTPS (the client side of TLS support)."""

    @pytest.mark.online
    def test_download_model_over_https(
        self, executable, server_port, timeouts, tmp_path
    ):
        """Download a tiny model from Hugging Face over HTTPS and serve it.

        Requires network access. Exercises the full client path including
        the hf.co -> CDN cross-host redirect. A fresh cache directory
        guarantees the download actually happens.
        """
        cache = tmp_path / "cache"
        cache.mkdir()
        args = base_exe_args(executable) + [
            "--server",
            "--port", str(server_port),
            "--hf-repo", HF_TEST_REPO,
            "--hf-file", HF_TEST_FILE,
        ]
        proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=fresh_cache_env(cache),
        )
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            ), "Server did not become ready (HTTPS download failed?)"

            if platform.system() != "Windows":
                # The downloaded weights landed in our fresh cache
                ggufs = [
                    p
                    for p in cache.rglob("*")
                    if p.is_file() and p.stat().st_size > 100_000
                ]
                assert ggufs, "No downloaded model found in fresh cache"
        finally:
            proc.terminate()
            proc.wait()

    def test_download_rejects_untrusted_cert(
        self, llamafile, executable, server_port, timeouts, ssl_cert, tmp_path
    ):
        """Downloading from a server with an untrusted (self-signed) cert fails.

        Runs fully offline: our own HTTPS server plays the untrusted host.
        Its /health endpoint would happily serve a body, so if the client's
        certificate verification were broken the download would 'succeed'
        and leave a file in the cache; instead the TLS handshake must fail
        and the cache must stay empty.
        """
        cert, key = ssl_cert
        rogue_port = server_port + 1
        server = llamafile.start_server(
            port=rogue_port,
            extra_args=["--ssl-cert-file", str(cert), "--ssl-key-file", str(key)],
        )
        try:
            assert wait_for_https_server(
                rogue_port, cert, timeout=timeouts.server_ready
            ), "TLS server did not become ready"

            cache = tmp_path / "cache"
            cache.mkdir()
            args = base_exe_args(executable) + [
                "--server",
                "--port", str(server_port),
                "--model-url", f"https://127.0.0.1:{rogue_port}/health",
            ]
            client = subprocess.run(
                args,
                capture_output=True,
                text=True,
                env=fresh_cache_env(cache),
                timeout=timeouts.server_ready,
            )
            assert client.returncode != 0, "Download from untrusted cert host succeeded"
            leftovers = [p for p in cache.rglob("*") if p.is_file()]
            assert not leftovers, (
                "Client fetched data from a server with an untrusted certificate: "
                f"{leftovers}"
            )
        finally:
            server.terminate()
            server.wait()
