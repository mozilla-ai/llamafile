#!/bin/sh
# Enable the llamafile plugin in Claude Code settings
#
# This script adds the llamafile plugin to .claude/settings.json
# It preserves existing settings and only adds what's needed.

set -e

SETTINGS_FILE=".claude/settings.json"

# Ensure .claude directory exists
mkdir -p .claude

# Check if jq is available for proper JSON handling
if command -v jq >/dev/null 2>&1; then
    if [ -f "$SETTINGS_FILE" ]; then
        # Merge with existing settings
        jq '.enabledPlugins["llamafile@llamafile-local"] = true |
            .extraKnownMarketplaces["llamafile-local"] = {
                "source": {
                    "source": "directory",
                    "path": ".claude/plugins/llamafile"
                }
            }' "$SETTINGS_FILE" > "$SETTINGS_FILE.tmp" && mv "$SETTINGS_FILE.tmp" "$SETTINGS_FILE"
        echo "Updated $SETTINGS_FILE with llamafile plugin"
    else
        # Create new settings file
        cat > "$SETTINGS_FILE" << 'EOF'
{
  "enabledPlugins": {
    "llamafile@llamafile-local": true
  },
  "extraKnownMarketplaces": {
    "llamafile-local": {
      "source": {
        "source": "directory",
        "path": ".claude/plugins/llamafile"
      }
    }
  }
}
EOF
        echo "Created $SETTINGS_FILE with llamafile plugin enabled"
    fi
else
    # No jq available - simple approach
    if [ -f "$SETTINGS_FILE" ]; then
        echo "Error: $SETTINGS_FILE exists but jq is not installed."
        echo "Please install jq to safely merge settings, or manually add:"
        echo ""
        echo '  "enabledPlugins": { "llamafile@llamafile-local": true }'
        echo '  "extraKnownMarketplaces": { "llamafile-local": { "source": { "source": "directory", "path": ".claude/plugins/llamafile" } } }'
        exit 1
    else
        cat > "$SETTINGS_FILE" << 'EOF'
{
  "enabledPlugins": {
    "llamafile@llamafile-local": true
  },
  "extraKnownMarketplaces": {
    "llamafile-local": {
      "source": {
        "source": "directory",
        "path": ".claude/plugins/llamafile"
      }
    }
  }
}
EOF
        echo "Created $SETTINGS_FILE with llamafile plugin enabled"
    fi
fi

echo ""
echo "Restart Claude Code to activate the plugin."
echo "Available commands: /build"
