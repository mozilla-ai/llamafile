# Security

llamafile can sandbox itself with two [Cosmopolitan
Libc](https://github.com/jart/cosmopolitan) primitives:

- **[pledge()](https://man.openbsd.org/pledge.2)** restricts which system
  calls the process may make. On Linux it installs a SECCOMP BPF filter;
  on OpenBSD it calls the native `pledge(2)`. **On by default.**
- **[unveil()](https://man.openbsd.org/unveil.2)** restricts which parts of
  the filesystem the process can see. On Linux it uses the
  [Landlock](https://docs.kernel.org/userspace-api/landlock.html) LSM
  (kernel 5.13+); on OpenBSD it calls the native `unveil(2)`. **Opt-in**,
  via `--confine-reads`.

Neither needs any kernel configuration or privileges. SECCOMP filtering
requires Linux 3.5+ on x86-64 and 5.13+ (for Landlock) on either
architecture; on older kernels, and on macOS, Windows, and other BSDs, the
calls are no-ops and llamafile logs that sandboxing is unavailable and keeps
running. The pledge() sandbox can be turned off entirely with `--unsecure`.

## The default: pledge()

The `llamafile --server` process runs under `stdio anet rpath` on Linux
(`stdio inet rpath` on OpenBSD). After startup this means:

- **No outbound network.** `anet` allows `accept()` but not `connect()`, so
  the only networking the server can do is answer connections it received.
  If the server is ever compromised (say, through a bug in the GGUF parser
  or an HTTP handler), this is the single most valuable restriction: it cuts
  the attacker's ability to exfiltrate anything over the network.
- **No writing, creating, deleting, executing, or forking.** A compromised
  server can't modify your files, drop a payload, or launch a program.
  (`--slot-save-path` and a prompt cache add write access to *their*
  directories only.)
- **Reads are allowed** (`rpath`) — anywhere the process could already read.
  This is deliberate: a server routinely opens files whose paths it only
  learns at request time (multimodal media, static assets), which cannot be
  known when a path allow-list would have to be locked. If you want to
  restrict *which* files are readable, see `--confine-reads` below.

`--cli` runs under `stdio rpath tty` and `--chat` under
`stdio rpath wpath cpath tty` — neither has any network access at all.

## Opt-in read confinement: `--confine-reads`

Passing `--confine-reads` to the server additionally applies `unveil()`,
confining reads to the executable and the directories holding the weights
(the model and its shards, `--mmproj`, LoRA adapters, a draft model, control
vectors, `--media-path`, a static `--path` web root) plus the two
name-resolution files a non-numeric `--host` needs. The rest of the
filesystem — `/etc/passwd`, your SSH keys, other users' files — becomes
invisible to the server even though it is world-readable.

Two things to know before relying on it:

- **It confines directories, not individual files.** The model's *parent
  directory* is unveiled (so multi-part GGUF shards beside it load), which
  means anything else in that directory is readable too. Put weights in a
  dedicated directory if you don't want their neighbours exposed.
- **It's incompatible with reading files by arbitrary path at request time.**
  Because the readable set is locked at startup, a request that references a
  file outside the unveiled directories will be denied. The built-in flows
  (media under `--media-path`, static files under `--path`) are covered;
  bespoke setups that read elsewhere are not.
- **It requires a filesystem Landlock can govern.** On the handful it cannot
  (some network mounts, virtiofs, 9p) llamafile keeps the pledge() sandbox
  but skips confinement rather than refusing to load the model, and logs
  that it did so.

## When the sandbox is relaxed or skipped

llamafile logs a notice in each case:

- **Outbound features** (`--rpc` for distributed inference, server-side
  tools, the MCP proxy) legitimately need to `connect()` out, so the
  networking promise is relaxed from `anet` to `inet` for them. Writes and
  exec stay blocked.
- **GPU mode gets no sandbox.** GPU backends are loaded dynamically and their
  drivers need device access (`ioctl`, `/dev/*`) that no promise set covers,
  so the sandbox is skipped whenever a GPU backend is loaded. **This is the
  common production case** — a GPU server is not sandboxed. Pass
  `--gpu disable` to force CPU inference with the sandbox.
- **Combined mode** (the default `llamafile -m model.gguf`, TUI + server in
  one process) hosts an in-process HTTP client that must `connect()` to the
  server, so it's skipped. Run `llamafile --server` for the sandboxed server.
- Model downloads (`-hf`, `--model-url`) happen *before* the sandbox is
  installed; once the download completes the process is sandboxed normally.

## A note on the penalty mode

On Linux, sandbox violations are configured to return `EPERM` (permission
denied) rather than killing the process, so a blocked syscall surfaces as an
ordinary I/O error. If a server refuses to start with a `pledge failed`
error (an exotic kernel or an outer seccomp policy), `--unsecure` disables
the sandbox. OpenBSD's native `pledge(2)` always terminates a violating
process with `SIGABRT` instead; that behavior is not configurable there.

## Verifying the sandbox

Unit tests exercise every promise set llamafile uses, assert that allowed
operations work while blocked ones fail, and check that `unveil()` confines
reads to the unveiled directory (run on Linux):

```sh
.cosmocc/4.0.2/bin/make o//tests/sandbox_test && o//tests/sandbox_test
```

Integration tests verify the running server end-to-end — every thread of the
server process carries the SECCOMP filter, completions still work inside the
sandbox, a bundled `/zip/` llamafile loads under it, `--confine-reads`
confines while the default does not, and `--unsecure` really disables it:

```sh
cd tests/integration
./run_tests.sh --executable ../../o/llamafile/llamafile \
    --model ../../models/TinyLLama-v0.1-5M-F16.gguf -m sandbox
```

You can also check a running server by hand:

```sh
llamafile --server -m model.gguf &
grep Seccomp /proc/$!/status   # "Seccomp: 2" means the filter is active
```

## Caveats

Your llamafile is able to protect itself against the outside world, but that
doesn't mean you're protected from llamafile. Sandboxing is self-imposed. If
you obtained your llamafile from an untrusted source then its author could
have simply modified it to not do that. In that case, you can run the
untrusted llamafile inside another sandbox, such as a virtual machine, to
make sure it behaves how you expect.
