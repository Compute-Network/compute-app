#!/bin/sh
# Compute Code installer — the AI coding agent, standalone.
# Usage: curl -fsSL https://computenetwork.sh/code.sh | sh
#
# Installs a single self-contained native binary (no Node.js required) to
# ~/.compute/bin/compute-code and adds it to your PATH. Run `compute-code`
# to start. If you also run a Compute node, `compute code` launches the same
# agent — this script is for installing just the agent on its own.
set -e

stty sane </dev/tty 2>/dev/null || true

REPO="Compute-Network/compute-code"
INSTALL_DIR="${COMPUTE_INSTALL_DIR:-$HOME/.compute/bin}"
BINARY_NAME="compute-code"

if [ -t 1 ] || [ -t 2 ]; then
  DIM='\033[2m'; BOLD='\033[1m'; GREEN='\033[32m'; RED='\033[31m'; RESET='\033[0m'
else
  DIM='' BOLD='' GREEN='' RED='' RESET=''
fi

spin() {
  _msg="$1"; _pid="$2"; _chars='/-\|'; _i=0
  while kill -0 "$_pid" 2>/dev/null; do
    _i=$(( (_i + 1) % 4 ))
    printf "\r  ${DIM}%s${RESET} %s" "$(echo "$_chars" | cut -c$((_i+1))-$((_i+1)))" "$_msg"
    sleep 0.1
  done
  wait "$_pid" 2>/dev/null; _code=$?
  if [ $_code -eq 0 ]; then printf "\r  ${GREEN}✓${RESET} %s\n" "$_msg"; else printf "\r  ${RED}✗${RESET} %s\n" "$_msg"; fi
  return $_code
}

# Detect OS
OS="$(uname -s)"
case "$OS" in
  Linux)  OS="linux"; OS_DISPLAY="Linux" ;;
  Darwin) OS="darwin"; OS_DISPLAY="macOS" ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "${RED}  Error: use the Windows installer instead${RESET}" >&2
    echo "${DIM}  https://github.com/${REPO}/releases/latest${RESET}" >&2
    exit 1 ;;
  *) echo "${RED}  Error: unsupported OS: $OS${RESET}" >&2; exit 1 ;;
esac

# Detect architecture
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64)  ARCH="x86_64"; ARCH_DISPLAY="x86_64" ;;
  arm64|aarch64) ARCH="aarch64"; ARCH_DISPLAY="ARM64" ;;
  *) echo "${RED}  Error: unsupported architecture: $ARCH${RESET}" >&2; exit 1 ;;
esac

case "$OS" in
  linux)  TARGET="${ARCH}-unknown-linux-gnu" ;;
  darwin) TARGET="${ARCH}-apple-darwin" ;;
esac

clear 2>/dev/null || true
echo ""
echo "  ${BOLD}C O M P U T E   C O D E${RESET}"
echo "  ${DIM}AI coding agent · powered by the Compute network${RESET}"
echo ""
echo "  ${DIM}Platform${RESET}  ${OS_DISPLAY} ${ARCH_DISPLAY}"

LATEST=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
if [ -z "$LATEST" ]; then
  echo ""
  echo "  ${RED}✗${RESET} Could not fetch latest release"
  echo "  ${DIM}See https://github.com/${REPO}/releases${RESET}"
  exit 1
fi
echo "  ${DIM}Version${RESET}   ${LATEST}"
echo ""

ASSET="${BINARY_NAME}-${TARGET}"
URL="https://github.com/${REPO}/releases/download/${LATEST}/${ASSET}"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

curl -fSL -o "${TMPDIR}/${ASSET}" "$URL" 2>/dev/null &
spin "Downloading Compute Code" $!

curl -fSL -o "${TMPDIR}/${ASSET}.sha256" "${URL}.sha256" 2>/dev/null || true
if [ -s "${TMPDIR}/${ASSET}.sha256" ]; then
  (
    cd "$TMPDIR"
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 -c "${ASSET}.sha256"
    elif command -v sha256sum >/dev/null 2>&1; then sha256sum -c "${ASSET}.sha256"
    fi
  ) >/dev/null &
  spin "Verifying checksum" $!
fi

mkdir -p "$INSTALL_DIR"
chmod +x "${TMPDIR}/${ASSET}"
mv "${TMPDIR}/${ASSET}" "${INSTALL_DIR}/${BINARY_NAME}"
echo "  ${GREEN}✓${RESET} Installed ${DIM}${BINARY_NAME}${RESET} to ${DIM}${INSTALL_DIR}${RESET}"

# Add to PATH if not already there
if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
  SHELL_NAME=$(basename "$SHELL")
  RC_FILE="$HOME/.${SHELL_NAME}rc"
  if [ -f "$RC_FILE" ] && grep -q "$INSTALL_DIR" "$RC_FILE" 2>/dev/null; then
    :
  else
    echo "export PATH=\"${INSTALL_DIR}:\$PATH\"" >> "$RC_FILE"
    echo "  ${GREEN}✓${RESET} Added to PATH ${DIM}(${RC_FILE})${RESET}"
  fi
fi

INSTALLED_VERSION=$("${INSTALL_DIR}/${BINARY_NAME}" --version 2>/dev/null || true)
if [ -z "$INSTALLED_VERSION" ]; then
  echo "  ${RED}✗${RESET} Could not run installed binary (--version failed)"
  echo "  ${DIM}Path: ${INSTALL_DIR}/${BINARY_NAME}${RESET}"
  exit 1
fi

echo ""
echo "  ${GREEN}${BOLD}Compute Code ${INSTALLED_VERSION} installed${RESET}"
echo ""
echo "  ${DIM}Start it anytime with${RESET} ${BOLD}compute-code${RESET}${DIM}.${RESET}"
echo ""

if [ -t 0 ]; then
  echo "  ${DIM}Launching${RESET} ${BOLD}${BINARY_NAME}${RESET}${DIM}...${RESET}"
  echo ""
  exec "${INSTALL_DIR}/${BINARY_NAME}"
else
  echo "  ${DIM}Run${RESET} ${BOLD}${BINARY_NAME}${RESET} ${DIM}to launch.${RESET}"
  echo ""
fi
