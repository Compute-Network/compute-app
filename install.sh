#!/bin/sh
# Compute CLI installer
# Usage: curl -fsSL https://computenetwork.sh/install.sh | sh
set -e

# If a prior installer / brew run left the terminal in raw-ish mode
# (cursor-down without carriage return, which makes the ASCII globe
# cascade diagonally), restore sane TTY settings before drawing anything.
# Under `curl ... | sh`, stdin is a pipe — not a TTY — so we redirect
# stty's file descriptor to /dev/tty (the controlling terminal) and
# swallow any failure for truly non-interactive contexts like CI.
stty sane </dev/tty 2>/dev/null || true

REPO="Compute-Network/compute-app"
INSTALL_DIR="${COMPUTE_INSTALL_DIR:-$HOME/.compute/bin}"
BINARY_NAME="compute"
STAGE_NODE_BINARY="llama_stage_tcp_node"
GATEWAY_BINARY="llama_stage_gateway_tcp_node"

# Colors
if [ -t 1 ] || [ -t 2 ]; then
  DIM='\033[2m'
  BOLD='\033[1m'
  GREEN='\033[32m'
  RED='\033[31m'
  RESET='\033[0m'
else
  DIM='' BOLD='' GREEN='' RED='' RESET=''
fi

# Spinner — \r-rewriting frame for tasks that finish in a second or two.
spin() {
  _spin_msg="$1"
  _spin_pid="$2"
  _spin_chars='/-\|'
  _spin_i=0
  while kill -0 "$_spin_pid" 2>/dev/null; do
    _spin_i=$(( (_spin_i + 1) % 4 ))
    printf "\r  ${DIM}%s${RESET} %s" "$(echo "$_spin_chars" | cut -c$((_spin_i+1))-$((_spin_i+1)))" "$_spin_msg"
    sleep 0.1
  done
  wait "$_spin_pid" 2>/dev/null
  _spin_code=$?
  if [ $_spin_code -eq 0 ]; then
    printf "\r  ${GREEN}✓${RESET} %s\n" "$_spin_msg"
  else
    printf "\r  ${RED}✗${RESET} %s\n" "$_spin_msg"
  fi
  return $_spin_code
}

# Long-running loader with elapsed-time counter. Same \r-rewriting line
# as spin(), but appends `(42s)` so the user can tell it's not hung.
# Pads the message on overwrite so characters from longer prior frames
# don't linger when the elapsed counter shortens — fixes the truncated-
# message garbage we'd see on brew installs.
loader() {
  _load_msg="$1"
  _load_pid="$2"
  _load_chars='/-\|'
  _load_i=0
  _load_start=$(date +%s 2>/dev/null || echo 0)
  _load_prev_len=0
  while kill -0 "$_load_pid" 2>/dev/null; do
    _load_i=$(( (_load_i + 1) % 4 ))
    _load_char=$(echo "$_load_chars" | cut -c$((_load_i+1))-$((_load_i+1)))
    _load_now=$(date +%s 2>/dev/null || echo 0)
    _load_elapsed=$(( _load_now - _load_start ))
    _load_render="  ${_load_char} ${_load_msg} (${_load_elapsed}s)"
    # Pad with trailing spaces to clear any remnant from longer frames.
    _load_pad=""
    _load_cur_len=${#_load_render}
    if [ "$_load_cur_len" -lt "$_load_prev_len" ]; then
      _load_diff=$(( _load_prev_len - _load_cur_len ))
      _load_pad=$(printf '%*s' "$_load_diff" '')
    fi
    _load_prev_len=$_load_cur_len
    printf "\r  ${DIM}%s${RESET} %s ${DIM}(%ds)${RESET}%s" \
      "$_load_char" "$_load_msg" "$_load_elapsed" "$_load_pad"
    sleep 0.2
  done
  wait "$_load_pid" 2>/dev/null
  _load_code=$?
  _load_now=$(date +%s 2>/dev/null || echo 0)
  _load_elapsed=$(( _load_now - _load_start ))
  # Clear the line fully before writing the final state.
  printf "\r%*s\r" 120 ''
  if [ "$_load_code" -eq 0 ]; then
    printf "  ${GREEN}✓${RESET} %s ${DIM}(%ds)${RESET}\n" "$_load_msg" "$_load_elapsed"
  else
    printf "  ${RED}✗${RESET} %s ${DIM}(%ds)${RESET}\n" "$_load_msg" "$_load_elapsed"
  fi
  return $_load_code
}


# ASCII globe
show_globe() {
  cat << 'GLOBE'
              _-o#&&*''''?d:>b\_
          _o/"`''  '',, dMF9MMMMMHo_
       .o&#'        `"MbHMMMMMMMMMMMHo.
     .o"" '         vodM*$&&HMMMMMMMMMM?.
    ,'              $M&ood,~'`(&##MMMMMMH\
   /               ,MMMMMMM#b?#bobMMMMHMMML
  &              ?MMMMMMMMMMMMMMMMM7MMM$R*Hk
 ?$.            :MMMMMMMMMMMMMMMMMMM/HMMM|`*L
|               |MMMMMMMMMMMMMMMMMMMMbMH'   T,
$H#:            `*MMMMMMMMMMMMMMMMMMMMb#}'  `?
]MMH#             ""*""""*#MMMMMMMMMMMMM'    -
MMMMMb_                   |MMMMMMMMMMMP'     :
HMMMMMMMHo                 `MMMMMMMMMT       .
?MMMMMMMMP                  9MMMMMMMM}       -
-?MMMMMMM                  |MMMMMMMMM?,d-    '
 :|MMMMMM-                 `MMMMMMMT .M|.   :
  .9MMM[                    &MMMMM*' `'    .
   :9MMk                    `MMM#"        -
     &M}                     `          .-
      `&.                             .
        `~,   .                     ./
            . _                  .-
              '`--._,dd###pp=""'
GLOBE
}

# Detect OS
OS="$(uname -s)"
case "$OS" in
  Linux)  OS="linux"; OS_DISPLAY="Linux" ;;
  Darwin) OS="darwin"; OS_DISPLAY="macOS" ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "${RED}  Error: use the Windows installer instead${RESET}" >&2
    echo "${DIM}  https://github.com/${REPO}/releases/latest${RESET}" >&2
    exit 1
    ;;
  *) echo "${RED}  Error: unsupported OS: $OS${RESET}" >&2; exit 1 ;;
esac

# Detect architecture
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64)  ARCH="x86_64"; ARCH_DISPLAY="x86_64" ;;
  arm64|aarch64) ARCH="aarch64"; ARCH_DISPLAY="ARM64" ;;
  *) echo "${RED}  Error: unsupported architecture: $ARCH${RESET}" >&2; exit 1 ;;
esac

# Build target triple
case "$OS" in
  linux)  TARGET="${ARCH}-unknown-linux-gnu" ;;
  darwin) TARGET="${ARCH}-apple-darwin" ;;
esac

# Show globe + header
clear 2>/dev/null || true
echo ""
echo "${DIM}$(show_globe)${RESET}"
echo ""
echo "  ${BOLD}C O M P U T E${RESET}"
echo "  ${DIM}Decentralized GPU Network${RESET}"
echo ""
echo "  ${DIM}Platform${RESET}  ${OS_DISPLAY} ${ARCH_DISPLAY}"

# Get latest release
LATEST=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')

if [ -z "$LATEST" ]; then
  echo ""
  echo "  ${RED}✗${RESET} Could not fetch latest release"
  echo ""
  echo "  ${DIM}Build from source:${RESET}"
  echo "  git clone https://github.com/${REPO}.git && cd compute-app"
  echo "  cargo build --release -p compute-cli"
  exit 1
fi

echo "  ${DIM}Version${RESET}   ${LATEST}"
echo ""

# Download in background
ASSET="compute-${TARGET}.tar.gz"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${LATEST}/${ASSET}"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

curl -fSL -o "${TMPDIR}/${ASSET}" "$DOWNLOAD_URL" 2>/dev/null &
spin "Downloading binary" $!

# Extract
tar xzf "${TMPDIR}/${ASSET}" -C "$TMPDIR" &
spin "Extracting archive" $!

mkdir -p "$INSTALL_DIR"

if [ ! -f "${TMPDIR}/compute" ]; then
  echo ""
  echo "  ${RED}✗${RESET} Archive did not contain compute binary"
  exit 1
fi

# Stop any running daemon and sidecars BEFORE swapping binaries on disk.
# Replacing the file alone is not enough: the kernel keeps the old inode
# alive for any process that already opened it, so the running gateway
# would keep serving the previous version's spec config (low max_k,
# missing instrumentation) until killed. The Compute TUI's "restart"
# button used to leak the gateway sidecar past daemon exit; install.sh
# now does the cleanup inline so a fresh `compute` run always picks up
# the binary it just unpacked.
PID_FILE="$HOME/.compute/compute.pid"
DAEMON_PID=""
if [ -f "$PID_FILE" ]; then
  DAEMON_PID=$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)
fi

(
  # SIGTERM the daemon first so its drop handlers run (graceful sidecar
  # cleanup, model unload).
  if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    kill -TERM "$DAEMON_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$DAEMON_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL "$DAEMON_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE" 2>/dev/null || true

  # Belt-and-suspenders: kill any orphaned sidecars by name. Match the
  # exact basename so we don't accidentally hit unrelated processes.
  for proc in llama_stage_gateway_tcp_node llama_stage_tcp_node; do
    if command -v pkill >/dev/null 2>&1; then
      pkill -TERM -x "$proc" 2>/dev/null || true
    else
      # macOS pkill is always present, but fall back to ps + kill for portability
      ps -A -o pid=,comm= 2>/dev/null | awk -v want="$proc" '$2 ~ want { print $1 }' | xargs -I {} kill -TERM {} 2>/dev/null || true
    fi
  done
  sleep 1
  for proc in llama_stage_gateway_tcp_node llama_stage_tcp_node; do
    if command -v pkill >/dev/null 2>&1; then
      pkill -KILL -x "$proc" 2>/dev/null || true
    else
      ps -A -o pid=,comm= 2>/dev/null | awk -v want="$proc" '$2 ~ want { print $1 }' | xargs -I {} kill -KILL {} 2>/dev/null || true
    fi
  done
) &
spin "Stopping running daemon and sidecars" $!

chmod +x "${TMPDIR}/compute"
mv "${TMPDIR}/compute" "${INSTALL_DIR}/${BINARY_NAME}"
echo "  ${GREEN}✓${RESET} Installed ${DIM}${BINARY_NAME}${RESET} to ${DIM}${INSTALL_DIR}${RESET}"

if [ -f "${TMPDIR}/${STAGE_NODE_BINARY}" ]; then
  chmod +x "${TMPDIR}/${STAGE_NODE_BINARY}"
  mv "${TMPDIR}/${STAGE_NODE_BINARY}" "${INSTALL_DIR}/${STAGE_NODE_BINARY}"
  echo "  ${GREEN}✓${RESET} Installed ${DIM}${STAGE_NODE_BINARY}${RESET}"
fi

if [ -f "${TMPDIR}/${GATEWAY_BINARY}" ]; then
  chmod +x "${TMPDIR}/${GATEWAY_BINARY}"
  mv "${TMPDIR}/${GATEWAY_BINARY}" "${INSTALL_DIR}/${GATEWAY_BINARY}"
  echo "  ${GREEN}✓${RESET} Installed ${DIM}${GATEWAY_BINARY}${RESET}"
fi

# Clean up any dylibs/sos from a previous install — otherwise stale libllama
# variants with different ABI versions can be mistakenly loaded and crash the
# stage nodes with missing @rpath errors.
for stale in "${INSTALL_DIR}"/*.dylib "${INSTALL_DIR}"/*.so "${INSTALL_DIR}"/*.so.*; do
  [ -e "$stale" ] || [ -L "$stale" ] || continue
  rm -f "$stale"
done

# Move any bundled shared libraries (libllama.dylib / libggml.dylib / .so)
# next to the binaries so the sidecars can dlopen them via @loader_path.
# -e fails for broken symlinks (whose targets we already moved earlier in the
# loop), so accept either an existing path or a symlink.
for libfile in "${TMPDIR}"/*.dylib "${TMPDIR}"/*.so "${TMPDIR}"/*.so.*; do
  [ -e "$libfile" ] || [ -L "$libfile" ] || continue
  base=$(basename "$libfile")
  mv "$libfile" "${INSTALL_DIR}/${base}"
  echo "  ${GREEN}✓${RESET} Installed ${DIM}${base}${RESET}"
done

# Add to PATH if not already there
if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
  SHELL_NAME=$(basename "$SHELL")
  RC_FILE="$HOME/.${SHELL_NAME}rc"
  EXPORT_LINE="export PATH=\"${INSTALL_DIR}:\$PATH\""

  if [ -f "$RC_FILE" ] && grep -q "$INSTALL_DIR" "$RC_FILE" 2>/dev/null; then
    : # Already in rc file
  else
    echo "$EXPORT_LINE" >> "$RC_FILE"
    echo "  ${GREEN}✓${RESET} Added to PATH ${DIM}(${RC_FILE})${RESET}"
  fi
fi

# v0.4.3: on Apple Silicon, install oMLX (https://github.com/jundot/omlx)
# — a Python + FastAPI MLX inference server compute-daemon uses to route
# MLX-format models (e.g. `qwen-3.6` → `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`)
# through Apple's MLX framework instead of llama-server's GGUF path.
#
# This is synchronous and visible: brew has its own download / install
# progress output (`==> Downloading ...`, percentage bars, etc), which
# is more informative than a bash spinner we'd have to fight for line
# control with. On a fresh Mac without Python this takes 2–5 minutes;
# subsequent installs (brew already has the deps) are seconds.
#
# Non-fatal: if brew is missing or the install fails, compute-daemon
# silently falls back to llama-server for MLX-preferred models.
OMLX_LOG="$HOME/.compute/logs/omlx-install.log"
if [ "$OS" = "darwin" ] && [ "$ARCH" = "aarch64" ]; then
  if command -v omlx >/dev/null 2>&1; then
    echo "  ${GREEN}✓${RESET} oMLX already installed ${DIM}($(command -v omlx))${RESET}"
  elif command -v brew >/dev/null 2>&1; then
    mkdir -p "$(dirname "$OMLX_LOG")"
    # Run brew inside `script -q /dev/null` so it writes to a fake PTY
    # instead of our real terminal. Without this, brew's native progress
    # output leaks past stdout/stderr redirects (it opens /dev/tty
    # directly for color + progress bars) and corrupts our spinner. The
    # subshell captures everything to a log file the user can tail if
    # they want the gritty details.
    (
      script -q /dev/null sh -c '
        brew tap jundot/omlx https://github.com/jundot/omlx 2>&1
        brew install jundot/omlx/omlx 2>&1
      '
    ) > "$OMLX_LOG" 2>&1 &
    loader "Installing oMLX (MLX inference server)" $! || {
      echo "     ${DIM}Log: ${OMLX_LOG}${RESET}"
      echo "     ${DIM}MLX models will use the GGUF path via llama-server. Retry: brew install jundot/omlx/omlx${RESET}"
    }
  else
    echo "  ${DIM}Note: Homebrew not found — MLX models will use the GGUF path via llama-server.${RESET}"
    echo "  ${DIM}      Install Homebrew (https://brew.sh) and re-run install.sh to enable MLX.${RESET}"
  fi
fi

# Verify the binary we just unpacked is actually executable and matches
# the release we intended to install. Confirms install.sh wrote real
# files (no silent permission errors) and surfaces the version so the
# user can see at a glance whether they're on the latest tag.
INSTALLED_VERSION=$("${INSTALL_DIR}/${BINARY_NAME}" --version 2>/dev/null || true)
if [ -n "$INSTALLED_VERSION" ]; then
  echo "  ${GREEN}✓${RESET} Verified ${DIM}${INSTALLED_VERSION}${RESET}"
else
  echo "  ${RED}✗${RESET} Could not run installed binary (--version failed)"
  echo "  ${DIM}Path: ${INSTALL_DIR}/${BINARY_NAME}${RESET}"
fi

echo ""
echo "  ${GREEN}${BOLD}Setup complete${RESET}"
echo ""
echo "  ${DIM}To launch Compute anytime, just type${RESET} ${BOLD}compute${RESET} ${DIM}into your terminal.${RESET}"
echo ""

# Only auto-launch when the installer was invoked directly (stdin is a tty).
# Under `curl ... | sh` stdin is a pipe and the shell's process group doesn't
# own the terminal, so even after redirecting to /dev/tty the launched TUI
# can't initialise its input reader — the user has to run it themselves.
if [ -t 0 ]; then
  echo "  ${DIM}Launching${RESET} ${BOLD}${BINARY_NAME}${RESET}${DIM}...${RESET}"
  echo ""
  exec "${INSTALL_DIR}/${BINARY_NAME}"
else
  echo "  ${DIM}Run${RESET} ${BOLD}${BINARY_NAME}${RESET} ${DIM}to launch.${RESET}"
  echo ""
fi
