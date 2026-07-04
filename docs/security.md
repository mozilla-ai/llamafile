# Security

llamafile sandboxes itself with two [Cosmopolitan
Libc](https://github.com/jart/cosmopolitan) primitives:

- **[pledge()](https://man.openbsd.org/pledge.2)** restricts which system
  calls the process may make. On Linux it installs a SECCOMP BPF filter;
  on OpenBSD it calls the native `pledge(2)`.
- **[unveil()](https://man.openbsd.org/unveil.2)** restricts which parts of
  the filesystem the process can see. On Linux it uses the
  [Landlock](https://docs.kernel.org/userspace-api/landlock.html) LSM
  (kernel 5.13+); on OpenBSD it calls the native `unveil(2)`.

Neither needs any kernel configuration or privileges. SECCOMP filtering
requires Linux 3.5+ on x86-64 and 5.13+ (for Landlock) on either
architecture; on older kernels, and on macOS, Windows, and other BSDs,
the calls are no-ops and llamafile logs that sandboxing is unavailable and
keeps running. Sandboxing is enabled by default and can be turned off with
the `--unsecure` flag.

## What each mode is allowed to do after startup

**HTTP server** (`llamafile --server`) runs under `stdio anet rpath` on
Linux (`stdio inet rpath` on OpenBSD):

- On Linux the only networking-related system call available after startup
  is `accept()` (that's what `anet` means — accept connections but never
  initiate them). If the server is ever compromised, this sharply limits
  an attacker's ability to exfiltrate data, because the process cannot open
  an outbound connection.
- Filesystem access is **read-only** (`rpath`), and `unveil()` confines
  those reads to the executable and the directories holding the weights
  (the model and its shards, an `--mmproj` file, LoRA adapters, a draft
  model, control vectors) plus the two name-resolution files a non-numeric
  `--host` needs. The rest of the filesystem — `/etc/passwd`, your SSH
  keys, other users' files — is invisible to the server even though it is
  world-readable. Writing, creating, deleting, executing programs, and
  forking are all denied. (`--slot-save-path` and a prompt cache
  additionally grant read-write-create access to their directories.)
- The read-only promise is needed even for a bundled llamafile whose
  weights are embedded as `/zip/model.gguf`: the loader reads them by
  reopening the executable's own zip store, which is a real `open()`.
- Path confinement (`unveil`) requires a filesystem the kernel's Landlock
  LSM can govern. On the handful it cannot (some network mounts, virtiofs,
  9p), llamafile keeps the pledge() syscall sandbox but skips path
  confinement rather than refusing to load the model, and says so in the
  log.

**CLI mode** (`llamafile --cli`) runs under `stdio rpath tty`: no network
access at all, enforced by the operating-system kernel, and no ability to
write to the filesystem. This keeps your computer safe in the event that a
bug is ever discovered in the GGUF file format that lets an attacker craft
malicious weights files and post them online.

**Chat mode** (`llamafile --chat`) runs under `stdio rpath wpath cpath
tty`: like CLI mode it has no network access, but it may create and write
files so interactive commands like `/dump` and `/upload` keep working.

## When the sandbox is not applied

llamafile prints a notice in each of these cases:

- **Combined mode** (the default, TUI + server in one process) hosts an
  in-process HTTP client that must `connect()` to the server, so the
  accept-only sandbox can't be used. Run `llamafile --server` if you want
  the sandboxed server.
- **GPU mode**: GPU backends are loaded dynamically and their drivers need
  device access that no reasonable promise set covers, so sandboxing is
  skipped when a GPU backend is loaded. Pass `--gpu disable` to force CPU
  inference with the sandbox.
- Model downloads (`-hf`, `--model-url`) happen *before* the sandbox is
  installed; once the download completes the process is sandboxed normally.

## A note on the penalty mode

On Linux, sandbox violations are configured to return `EPERM` (permission
denied) rather than killing the process, so a blocked syscall surfaces as
an ordinary I/O error. OpenBSD's native `pledge(2)` always terminates a
violating process with `SIGABRT` instead; that behavior is not
configurable there.

## Verifying the sandbox

Unit tests exercise every promise set llamafile uses, assert that allowed
operations work while blocked ones fail, and check that `unveil()` confines
reads to the unveiled directory (run on Linux):

```sh
.cosmocc/4.0.2/bin/make o//tests/sandbox_test && o//tests/sandbox_test
```

Integration tests verify the running server end-to-end — every thread of
the server process carries the SECCOMP filter, completions still work
inside the sandbox, a bundled `/zip/` llamafile loads under it, and
`--unsecure` really disables it:

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

Your llamafile is able to protect itself against the outside world, but
that doesn't mean you're protected from llamafile. Sandboxing is
self-imposed. If you obtained your llamafile from an untrusted source then
its author could have simply modified it to not do that. In that case, you
can run the untrusted llamafile inside another sandbox, such as a virtual
machine, to make sure it behaves how you expect.
