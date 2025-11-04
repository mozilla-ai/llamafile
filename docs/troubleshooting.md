# Troubleshooting

This page covers common issues and their solutions when using llamafile.

## All Platforms

### CrowdStrike blocking llamafile

On any platform, if your llamafile process is immediately killed, check if you have CrowdStrike and then ask to be whitelisted.

## macOS

### Xcode Command Line Tools required

On macOS with Apple Silicon you need to have **Xcode Command Line Tools** installed for llamafile to be able to bootstrap itself and use GPU acceleration.

### Shell compatibility issues

If you use zsh and have trouble running llamafile, try saying `sh -c ./llamafile`. This is due to a bug that was fixed in zsh 5.9+. The same is the case for Python `subprocess`, old versions of Fish, etc.

### "Cannot be opened because the developer cannot be verified"

If you see this error, you have two options:

**Option 1: Use System Settings (Recommended)**

1. Immediately launch System Settings, then go to Privacy & Security
2. llamafile should be listed at the bottom, with a button to Allow

**Option 2: Temporarily disable Gatekeeper**

If the file isn't listed in System Settings, change your command in the Terminal to:

```sh
sudo spctl --master-disable; [llama launch command]; sudo spctl --master-enable
```

!!! warning
    `--master-disable` disables _all_ checking, so you need to turn it back on after quitting llama.

## Linux

### `run-detectors` or WINE errors

On some Linux systems, you might get errors relating to `run-detectors` or WINE. This is due to `binfmt_misc` registrations. You can fix that by adding an additional registration for the APE file format llamafile uses:

```sh
sudo wget -O /usr/bin/ape https://cosmo.zip/pub/cosmos/bin/ape-$(uname -m).elf
sudo chmod +x /usr/bin/ape
sudo sh -c "echo ':APE:M::MZqFpD::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"
sudo sh -c "echo ':APE-jart:M::jartsr::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"
```

## Windows

### File size limit

Windows has a maximum file size limit of **4GB** for executables. The LLaVA server executable is just 30MB shy of that limit, so it'll work on Windows, but with larger models like WizardCoder 13B, you need to store the weights in a separate file. See [Using llamafile with external weights](usage.md#using-llamafile-with-external-weights).

### Renaming to .exe

On Windows, you may need to rename your llamafile by adding `.exe` to the filename for it to be recognized as an executable.

## WSL (Windows Subsystem for Linux)

### General WSL setup

On WSL, there are many possible gotchas. One thing that helps solve them completely is creating a systemd service for the APE loader.

Put the following in `/etc/systemd/system/cosmo-binfmt.service`:

```ini
[Unit]
Description=cosmopolitan APE binfmt service
After=wsl-binfmt.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c "echo ':APE:M::MZqFpD::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"

[Install]
WantedBy=multi-user.target
```

Ensure that the APE loader is installed to `/usr/bin/ape`:

```sh
sudo wget -O /usr/bin/ape https://cosmo.zip/pub/cosmos/bin/ape-$(uname -m).elf
sudo chmod +x /usr/bin/ape
```

Then run:

```sh
sudo systemctl enable --now cosmo-binfmt
```

### Disabling WIN32 interop

Another thing that's helped WSL users who experience issues, is to disable the WIN32 interop feature:

```sh
sudo sh -c "echo -1 > /proc/sys/fs/binfmt_misc/WSLInterop"
```

In Windows 11 with WSL 2 the location of the interop flag has changed, as such the following command may be required instead/additionally:

```sh
sudo sh -c "echo -1 > /proc/sys/fs/binfmt_misc/WSLInterop-late"
```

### Permanently disabling interop

In the instance of getting a `Permission Denied` on disabling interop through CLI, it can be permanently disabled by adding the following in `/etc/wsl.conf`:

```ini
[interop]
enabled=false
```
