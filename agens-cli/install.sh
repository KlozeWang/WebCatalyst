#!/usr/bin/env bash
# agens installer
# - Detects Node ≥ 24 (required: Copilot SDK runtime imports node:sqlite)
# - If missing, downloads a portable Node into ~/.local/node
# - Installs the agens tarball globally (or links it during dev)
#
# Usage:
#   bash install.sh                  # install from sibling agens-*.tgz
#   bash install.sh /path/to/tgz     # install from given tarball
#   bash install.sh --link           # `npm link` (dev mode) from this directory
#
# Honors:
#   AGENS_NODE_VERSION    default 24.2.0
#   AGENS_NODE_PREFIX     default $HOME/.local/node
set -euo pipefail

NODE_VERSION="${AGENS_NODE_VERSION:-24.2.0}"
NODE_PREFIX="${AGENS_NODE_PREFIX:-$HOME/.local/node}"
MIN_MAJOR=24

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARBALL=""
DEV_LINK=0

if [ "${1:-}" = "--link" ]; then
  DEV_LINK=1
elif [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
elif [ -n "${1:-}" ]; then
  TARBALL="$1"
fi

c_green() { printf '\033[32m%s\033[0m\n' "$*"; }
c_yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
c_red() { printf '\033[31m%s\033[0m\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- Detect Node ---
node_ok=0
if have node; then
  major=$(node -e 'console.log(process.versions.node.split(".")[0])' 2>/dev/null || echo 0)
  if [ "$major" -ge "$MIN_MAJOR" ] 2>/dev/null; then
    node_ok=1
    c_green "✓ Node $(node --version) detected (≥ v${MIN_MAJOR})."
  else
    c_yellow "⚠ Node $(node --version) is older than v${MIN_MAJOR}. Will install a private copy."
  fi
fi

# --- Install portable Node if needed ---
if [ "$node_ok" -eq 0 ]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  NODE_ARCH="linux-x64" ;;
    Linux-aarch64) NODE_ARCH="linux-arm64" ;;
    Darwin-x86_64) NODE_ARCH="darwin-x64" ;;
    Darwin-arm64)  NODE_ARCH="darwin-arm64" ;;
    *) c_red "Unsupported platform $(uname -s)-$(uname -m). Please install Node ${MIN_MAJOR}+ manually."; exit 1 ;;
  esac

  TARFILE="node-v${NODE_VERSION}-${NODE_ARCH}.tar.xz"
  URL="https://nodejs.org/dist/v${NODE_VERSION}/${TARFILE}"
  TMP=$(mktemp -d)
  c_yellow "→ Downloading Node v${NODE_VERSION} (${NODE_ARCH})…"
  curl -fL --progress-bar "$URL" -o "$TMP/$TARFILE"

  mkdir -p "$(dirname "$NODE_PREFIX")"
  rm -rf "$NODE_PREFIX"
  tar -xJf "$TMP/$TARFILE" -C "$(dirname "$NODE_PREFIX")"
  mv "$(dirname "$NODE_PREFIX")/node-v${NODE_VERSION}-${NODE_ARCH}" "$NODE_PREFIX"
  rm -rf "$TMP"

  export PATH="$NODE_PREFIX/bin:$PATH"
  c_green "✓ Installed Node $("$NODE_PREFIX/bin/node" --version) into $NODE_PREFIX"

  # Persist PATH for future shells
  for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    [ -f "$rc" ] || continue
    if ! grep -qs "$NODE_PREFIX/bin" "$rc"; then
      printf '\n# Added by agens installer\nexport PATH="%s/bin:$PATH"\n' "$NODE_PREFIX" >> "$rc"
      c_green "✓ Added Node to PATH in $rc"
    fi
  done
fi

# --- Install agens ---
if [ "$DEV_LINK" -eq 1 ]; then
  c_yellow "→ Dev install: npm link from $SCRIPT_DIR"
  ( cd "$SCRIPT_DIR" && npm install && npm run build && npm link )
else
  if [ -z "$TARBALL" ]; then
    TARBALL=$(ls "$SCRIPT_DIR"/agens-*.tgz 2>/dev/null | head -n1 || true)
    if [ -z "$TARBALL" ]; then
      c_red "No agens-*.tgz found next to install.sh. Pass a path: bash install.sh /path/to/agens-x.y.z.tgz"
      exit 1
    fi
  fi
  c_yellow "→ Installing $TARBALL globally…"
  npm install -g "$TARBALL"
fi

# --- Verify ---
if have agens; then
  c_green ""
  c_green "✓ agens installed: $(agens --version)"
  c_green ""
  c_green "Try it:"
  c_green "  agens \"make a smiling-sun SVG landing page\""
  c_green ""
  c_green "Configure (optional — defaults work out of the box):"
  c_green "  agens config show"
  c_green "  agens config set provider.baseUrl <url>"
else
  c_yellow "agens installed, but not in PATH yet. Open a new shell or:"
  c_yellow "  export PATH=\"$NODE_PREFIX/bin:\$PATH\""
fi
