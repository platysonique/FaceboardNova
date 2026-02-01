#!/bin/bash
# Comprehensive installation script for Faceboard Nova
# Handles all dependencies, setup, and command installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Faceboard Nova Installation         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""

# Get the directory where this script is located
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "${BLUE}Installation directory:${NC} $INSTALL_DIR"

# Check if the main application file exists
if [ ! -f "$INSTALL_DIR/ar_face_masks_gui.py" ]; then
    echo -e "${RED}✗ Error: ar_face_masks_gui.py not found in $INSTALL_DIR${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 1: Checking Python installation...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}✗ Python 3.8 or higher is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check pip
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo -e "${YELLOW}⚠ pip3 not found, installing...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y python3-pip
    else
        echo -e "${RED}✗ Please install pip3 manually${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ pip3 found${NC}"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 2: Setting up virtual environment...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$INSTALL_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --quiet --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 3: Installing Python dependencies...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check which packages are already installed
REQUIRED_PACKAGES=("opencv-python" "numpy" "mediapipe" "Pillow" "PyQt5" "pyvirtualcam")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show "$package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All Python dependencies are already installed${NC}"
else
    echo "Installing missing packages: ${MISSING_PACKAGES[*]}"
    echo "(This may take a few minutes...)"
    pip install --quiet "${MISSING_PACKAGES[@]}"
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 4: Checking system dependencies...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check for v4l2loopback
if ! modinfo v4l2loopback &> /dev/null; then
    echo -e "${YELLOW}⚠ v4l2loopback kernel module not found${NC}"
    echo ""
    echo "Virtual camera support requires v4l2loopback."
    echo "Would you like to install it now? (requires sudo) [y/N]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        if command -v apt-get &> /dev/null; then
            echo "Installing v4l2loopback..."
            sudo apt-get update
            sudo apt-get install -y v4l2loopback-dkms v4l-utils
            echo -e "${GREEN}✓ v4l2loopback installed${NC}"
        else
            echo -e "${YELLOW}⚠ Please install v4l2loopback manually for your distribution${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Skipping v4l2loopback installation${NC}"
        echo "You can install it later with: sudo apt-get install v4l2loopback-dkms v4l-utils"
    fi
else
    echo -e "${GREEN}✓ v4l2loopback kernel module found${NC}"
fi

# Check for v4l-utils
if ! command -v v4l2-ctl &> /dev/null; then
    echo -e "${YELLOW}⚠ v4l-utils not found${NC}"
    if command -v apt-get &> /dev/null; then
        echo "Installing v4l-utils..."
        sudo apt-get install -y v4l-utils
        echo -e "${GREEN}✓ v4l-utils installed${NC}"
    fi
else
    echo -e "${GREEN}✓ v4l-utils found${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 5: Setting up virtual camera devices...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if v4l2loopback devices exist
DEVICES_EXIST=true
for i in 9 10 11 12 13; do
    if [ ! -e "/dev/video$i" ]; then
        DEVICES_EXIST=false
        break
    fi
done

if [ "$DEVICES_EXIST" = false ]; then
    echo -e "${YELLOW}⚠ Virtual camera devices not found${NC}"
    echo ""
    echo "Faceboard Nova needs virtual camera devices to work with OBS."
    echo "Would you like to set them up now? (requires sudo) [y/N]"
    echo "Note: OBS Virtual Camera must be stopped first!"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        if [ -f "$INSTALL_DIR/fix_vcam_conflict.sh" ]; then
            echo "Running virtual camera setup script..."
            bash "$INSTALL_DIR/fix_vcam_conflict.sh"
        else
            echo -e "${YELLOW}⚠ fix_vcam_conflict.sh not found, setting up manually...${NC}"
            echo "Unloading existing v4l2loopback module..."
            sudo modprobe -r v4l2loopback 2>/dev/null || true
            echo "Loading v4l2loopback with 5 devices..."
            sudo modprobe v4l2loopback devices=5 video_nr=9,10,11,12,13 card_label="cumcam","VCam-1","VCam-2","VCam-3","VCam-4" exclusive_caps=1
            echo -e "${GREEN}✓ Virtual camera devices created${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Skipping virtual camera setup${NC}"
        echo "You can set it up later with: bash fix_vcam_conflict.sh"
    fi
else
    echo -e "${GREEN}✓ Virtual camera devices already exist${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 6: Creating 'fbnova' command...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Determine where to install the command
if [ -w "$HOME/.local/bin" ] || mkdir -p "$HOME/.local/bin" 2>/dev/null; then
    BIN_DIR="$HOME/.local/bin"
    USE_SUDO=false
elif [ -w "/usr/local/bin" ]; then
    BIN_DIR="/usr/local/bin"
    USE_SUDO=false
else
    BIN_DIR="/usr/local/bin"
    USE_SUDO=true
fi

echo "Installing 'fbnova' command to: $BIN_DIR"

# Create the wrapper script
WRAPPER_SCRIPT="$BIN_DIR/fbnova"
cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
# Faceboard Nova launcher
# This script launches Faceboard Nova from any directory

INSTALL_DIR="INSTALL_DIR_PLACEHOLDER"

# Change to installation directory
cd "$INSTALL_DIR" || {
    echo "Error: Could not find Faceboard Nova installation directory"
    echo "Expected: $INSTALL_DIR"
    exit 1
}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the GUI application
python3 ar_face_masks_gui.py "$@"
EOF

# Replace the placeholder with actual install directory
sed -i "s|INSTALL_DIR_PLACEHOLDER|$INSTALL_DIR|g" "$WRAPPER_SCRIPT"

# Make the script executable
chmod +x "$WRAPPER_SCRIPT"

echo -e "${GREEN}✓ Created $WRAPPER_SCRIPT${NC}"

# Check if the bin directory is in PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo ""
    echo -e "${YELLOW}⚠ Warning: $BIN_DIR is not in your PATH${NC}"
    echo ""
    echo "To use 'fbnova' from anywhere, add this to your ~/.bashrc or ~/.zshrc:"
    echo -e "${GREEN}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
    echo ""
    
    # Offer to add it automatically
    if [ "$BIN_DIR" = "$HOME/.local/bin" ]; then
        echo "Would you like to add it automatically? [y/N]"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            SHELL_RC=""
            if [ -f "$HOME/.bashrc" ]; then
                SHELL_RC="$HOME/.bashrc"
            elif [ -f "$HOME/.zshrc" ]; then
                SHELL_RC="$HOME/.zshrc"
            fi
            
            if [ -n "$SHELL_RC" ]; then
                if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$SHELL_RC"; then
                    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
                    echo -e "${GREEN}✓ Added to $SHELL_RC${NC}"
                    echo "Run 'source $SHELL_RC' or restart your terminal to use 'fbnova'"
                else
                    echo -e "${GREEN}✓ Already in $SHELL_RC${NC}"
                fi
            fi
        fi
    fi
else
    echo -e "${GREEN}✓ $BIN_DIR is already in your PATH${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 7: Creating masks directory...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Create masks directory if it doesn't exist
if [ ! -d "$INSTALL_DIR/masks" ]; then
    mkdir -p "$INSTALL_DIR/masks"
    echo -e "${GREEN}✓ Created masks directory${NC}"
    echo -e "${YELLOW}⚠ Remember: Faceboard Nova does NOT come with masks${NC}"
    echo "  Add your own mask images (PNG, JPG, or GIF) to: $INSTALL_DIR/masks/"
else
    echo -e "${GREEN}✓ Masks directory already exists${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Installation Complete! 🎉           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "You can now run Faceboard Nova with:"
echo -e "${GREEN}  fbnova${NC}"
echo ""
echo "If the command is not found:"
echo "  1. Make sure $BIN_DIR is in your PATH"
echo "  2. Restart your terminal or run: source ~/.bashrc"
echo ""
echo "Next steps:"
echo "  1. Add mask images to: $INSTALL_DIR/masks/"
echo "  2. Run: fbnova"
echo "  3. Click 'Start' and assign masks to faces"
echo ""
echo -e "${BLUE}Enjoy Faceboard Nova! 🎭${NC}"
