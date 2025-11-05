llamafile adds pledge() and SECCOMP sandboxing to llama.cpp. This is
enabled by default. It can be turned off by passing the `--unsecure`
flag. Sandboxing is currently only supported on Linux and OpenBSD on
systems without GPUs; on other platforms it'll simply log a warning.

Our approach to security has these benefits:

1. After it starts up, your HTTP server isn't able to access the
   filesystem at all. This is good, since it means if someone discovers
   a bug in the llama.cpp server, then it's much less likely they'll be
   able to access sensitive information on your machine or make changes
   to its configuration. On Linux, we're able to sandbox things even
   further; the only networking related system call the HTTP server will
   allowed to use after starting up, is accept(). That further limits an
   attacker's ability to exfiltrate information, in the event that your
   HTTP server is compromised.

2. The main CLI command won't be able to access the network at all. This
   is enforced by the operating system kernel. It also won't be able to
   write to the file system. This keeps your computer safe in the event
   that a bug is ever discovered in the GGUF file format that lets
   an attacker craft malicious weights files and post them online. The
   only exception to this rule is if you pass the `--prompt-cache` flag
   without also specifying `--prompt-cache-ro`. In that case, security
   currently needs to be weakened to allow `cpath` and `wpath` access,
   but network access will remain forbidden.

Therefore your llamafile is able to protect itself against the outside
world, but that doesn't mean you're protected from llamafile. Sandboxing
is self-imposed. If you obtained your llamafile from an untrusted source
then its author could have simply modified it to not do that. In that
case, you can run the untrusted llamafile inside another sandbox, such
as a virtual machine, to make sure it behaves how you expect.
