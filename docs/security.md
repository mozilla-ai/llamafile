# Security

llamafile sandboxes itself with [pledge()](https://man.openbsd.org/pledge.2)
syscall restrictions, provided by Cosmopolitan Libc. On Linux the promises
are enforced with a SECCOMP BPF filter installed by the process itself (no
kernel configuration, containers, or privileges required; works on x86-64
and aarch64, kernel 2.6.23 or later). On OpenBSD the native `pledge(2)` is
used. Sandboxing is enabled by default and can be turned off with the
`--unsecure` flag. On platforms where it can't be enforced (macOS, Windows,
other BSDs) llamafile logs that sandboxing is unavailable and keeps running.

What each execution mode is allowed to do after startup:

1. **HTTP server** (`llamafile --server`) runs under `stdio anet`
   (Linux) or `stdio inet` (OpenBSD). On Linux the only networking
   related system call available to the server after startup is
   `accept()`: it can answer incoming requests but can never *initiate*
   a connection, which limits an attacker's ability to exfiltrate data
   if the server is ever compromised. Filesystem access is read-only
   (`rpath`), and only granted at all when something actually has to be
   read from disk after startup (the model weights, an `--mmproj` file,
   LoRA adapters, or a `--path` web root); a llamafile with embedded
   weights runs with no filesystem access whatsoever, since ZipOS reads
   are served from the already-mapped executable. Writing to the
   filesystem, executing programs, and forking are always denied
   (exception: `--slot-save-path` adds `wpath cpath` so slot saving can
   work). The sandbox is installed *before* the HTTP listener threads
   are created and *before* the (potentially untrusted) GGUF file is
   parsed.

2. **CLI mode** (`llamafile --cli`) runs under `stdio rpath tty`: no
   network access at all, enforced by the operating system kernel, and
   no ability to write to the filesystem. This keeps your computer safe
   in the event that a bug is ever discovered in the GGUF file format
   that lets an attacker craft malicious weights files and post them
   online.

3. **Chat mode** (`llamafile --chat`) runs under
   `stdio rpath wpath cpath tty`: like CLI mode it has no network
   access, but it may create and write files so interactive commands
   like `/dump` and `/upload` keep working.

Cases where the sandbox is not applied (llamafile prints a notice in
each of them):

- **Combined mode** (the default, TUI + server in one process) hosts an
  in-process HTTP client that must `connect()` to the server, so the
  accept-only sandbox can't be used. Run `llamafile --server` if you
  want the sandboxed server.
- **GPU mode**: GPU backends are loaded dynamically and drivers need
  device access that no reasonable promise set covers, so sandboxing is
  skipped when a GPU backend is loaded (as in previous llamafile
  releases). Pass `--gpu disable` to force CPU inference with the
  sandbox.
- **Router mode** (server without a model) spawns one child server
  process per model, which requires `fork`/`exec`; the router itself is
  not sandboxed, but every child server it spawns is.
- Model downloads (`-hf`, `--model-url`) happen *before* the sandbox is
  installed; once the download completes the process is sandboxed
  normally.

Violations are configured to return `EPERM` (permission denied) rather
than killing the process, so a blocked syscall surfaces as an ordinary
I/O error.

## Verifying the sandbox

Unit tests exercise every promise set llamafile uses and assert that
allowed operations work while blocked ones fail (run on Linux):

```sh
.cosmocc/4.0.2/bin/make o//tests/sandbox_test && o//tests/sandbox_test
```

Integration tests verify the running server end-to-end — the kernel
reports one extra SECCOMP filter on the server process
(`Seccomp: 2` / `Seccomp_filters` in `/proc/<pid>/status`), completions
still work inside the sandbox, and `--unsecure` really disables it:

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
self-imposed. If you obtained your llamafile from an untrusted source
then its author could have simply modified it to not do that. In that
case, you can run the untrusted llamafile inside another sandbox, such
as a virtual machine, to make sure it behaves how you expect.
