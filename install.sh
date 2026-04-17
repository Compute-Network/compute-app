#!/bin/sh
# Compute CLI installer
# Usage: curl -fsSL https://computenetwork.sh/install.sh | sh
set -e

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

# Spinner
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
