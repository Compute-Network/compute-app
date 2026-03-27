#!/bin/bash
set -euo pipefail

# Compute CLI Installer
# Usage: curl -fsSL https://computenetwork.sh/install.sh | bash

REPO="Compute-Network/compute-app"
BINARY_NAME="compute"
INSTALL_DIR="/usr/local/bin"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
DIM='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m'

info() { printf "${DIM}%s${NC}\n" "$1"; }
success() { printf "${GREEN}✓${NC} %s\n" "$1"; }
error() { printf "${RED}✗ %s${NC}\n" "$1"; exit 1; }

# --- Detect platform ---
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)   PLATFORM="linux" ;;
  Darwin)  PLATFORM="darwin" ;;
  MINGW*|MSYS*|CYGWIN*) error "Use install.ps1 for Windows" ;;
  *)       error "Unsupported OS: $OS" ;;
esac

case "$ARCH" in
  x86_64)        ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *)             error "Unsupported architecture: $ARCH" ;;
esac

FILENAME="compute-${PLATFORM}-${ARCH}.tar.gz"

echo ""
printf "  ${BOLD}██████╗ ██████╗ ███╗   ███╗██████╗ ██╗   ██╗████████╗███████╗${NC}\n"
printf "  ${BOLD}██╔════╝██╔═══██╗████╗ ████║██╔══██╗██║   ██║╚══██╔══╝██╔════╝${NC}\n"
printf "  ${BOLD}██║     ██║   ██║██╔████╔██║██████╔╝██║   ██║   ██║   █████╗  ${NC}\n"
printf "  ${BOLD}██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██║   ██║   ██║   ██╔══╝  ${NC}\n"
printf "  ${BOLD}╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ╚██████╔╝   ██║   ███████╗${NC}\n"
printf "  ${BOLD} ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝      ╚═════╝    ╚═╝   ╚══════╝${NC}\n"
echo ""
info "  Decentralized GPU Infrastructure"
echo ""

# --- Get latest release ---
info "  Fetching latest release..."

LATEST_URL="https://api.github.com/repos/${REPO}/releases/latest"
RELEASE_DATA=$(curl -fsSL "$LATEST_URL" 2>/dev/null) || error "Failed to fetch release info. Is the repo public?"

DOWNLOAD_URL=$(echo "$RELEASE_DATA" | grep -o "https://[^\"]*${FILENAME}" | head -1)

if [ -z "$DOWNLOAD_URL" ]; then
  # Fallback: try the tag-based URL
  TAG=$(echo "$RELEASE_DATA" | grep '"tag_name"' | head -1 | sed 's/.*: "//;s/".*//')
  if [ -n "$TAG" ]; then
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${TAG}/${FILENAME}"
  else
    error "Could not find release for ${PLATFORM}-${ARCH}"
  fi
fi

info "  Downloading ${FILENAME}..."

# --- Download & install ---
TMPDIR=$(mktemp -d)
trap "rm -rf '$TMPDIR'" EXIT

curl -fsSL "$DOWNLOAD_URL" -o "${TMPDIR}/${FILENAME}" || error "Download failed"
tar -xzf "${TMPDIR}/${FILENAME}" -C "$TMPDIR" || error "Extract failed"

if [ ! -f "${TMPDIR}/${BINARY_NAME}" ]; then
  error "Binary not found in archive"
fi

# Try system install dir, fall back to user dir
if [ -w "$INSTALL_DIR" ]; then
  mv "${TMPDIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
  chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
  success "Installed to ${INSTALL_DIR}/${BINARY_NAME}"
elif command -v sudo >/dev/null 2>&1; then
  info "  Installing to ${INSTALL_DIR} (requires sudo)..."
  sudo mv "${TMPDIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
  sudo chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
  success "Installed to ${INSTALL_DIR}/${BINARY_NAME}"
else
  # Fall back to ~/.compute/bin
  INSTALL_DIR="${HOME}/.compute/bin"
  mkdir -p "$INSTALL_DIR"
  mv "${TMPDIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
  chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
  success "Installed to ${INSTALL_DIR}/${BINARY_NAME}"

  # Check if in PATH
  case ":$PATH:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
      echo ""
      info "  Add this to your shell profile:"
      echo "    export PATH=\"\$HOME/.compute/bin:\$PATH\""
      ;;
  esac
fi

echo ""

# --- Verify ---
if command -v compute >/dev/null 2>&1; then
  VERSION=$(compute --version 2>/dev/null || echo "unknown")
  success "Compute CLI ${VERSION}"
else
  success "Installed successfully"
fi

echo ""
info "  Get started:"
echo "    compute init        First-time setup"
echo "    compute start       Start contributing compute"
echo "    compute dashboard   View live stats"
echo ""
