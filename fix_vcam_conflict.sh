#!/bin/bash
# Fix virtual camera conflict between Faceboard Nova and OBS
# Based on research: OBS Virtual Camera ALWAYS uses the FIRST v4l2loopback device
# Solution: Keep video9 for OBS, create video10-13 for Faceboard Nova

echo "=== Fixing Virtual Camera Conflict ==="
echo ""
echo "RESEARCH FINDINGS:"
echo "  - OBS Virtual Camera automatically uses the FIRST v4l2loopback device"
echo "  - OBS cannot be configured to use a different device"
echo "  - Solution: Keep video9 for OBS, use video10+ for Faceboard Nova"
echo ""

# Check current devices
echo "Current v4l2loopback devices:"
v4l2-ctl --list-devices 2>/dev/null | grep -A 1 "v4l2loopback" || echo "  None found"

echo ""
echo "Checking device status..."
video9_exists=0
video9_is_v4l2=0
if [ -e "/dev/video9" ]; then
    video9_exists=1
    echo "  ✓ /dev/video9 exists"
    if v4l2-ctl --device /dev/video9 --info 2>/dev/null | grep -q "v4l2loopback"; then
        video9_is_v4l2=1
        echo "  ✓ /dev/video9 is v4l2loopback (OBS will use this)"
    fi
else
    echo "  ✗ /dev/video9 MISSING"
fi

missing_devices=0
for i in 10 11 12 13; do
    if [ ! -e "/dev/video$i" ]; then
        missing_devices=1
        echo "  ✗ /dev/video$i MISSING"
    else
        echo "  ✓ /dev/video$i exists"
    fi
done

if [ $missing_devices -eq 0 ] && [ $video9_exists -eq 1 ]; then
    echo ""
    echo "✓ All devices exist! You're all set."
    echo ""
    echo "Available devices:"
    v4l2-ctl --list-devices 2>/dev/null | grep -A 1 "VCam\|v4l2loopback\|video"
    echo ""
    echo "Restart Faceboard Nova for it to detect the devices."
    exit 0
fi

echo ""
echo "Creating missing v4l2loopback devices..."
echo "  (This requires sudo - you'll be prompted for your password)"
echo ""

# Check if v4l2loopback is already loaded
if lsmod | grep -q v4l2loopback; then
    echo "  v4l2loopback module is already loaded"
    echo ""
    echo "  ⚠ CRITICAL: OBS Virtual Camera MUST be stopped before reloading!"
    echo "  If OBS is using video9, unloading will fail."
    echo ""
    echo "  Unloading module..."
    sudo modprobe -r v4l2loopback 2>&1
    if [ $? -ne 0 ]; then
        echo ""
        echo "⚠ Failed to unload v4l2loopback module"
        echo "  This means a device is currently in use (likely OBS Virtual Camera)"
        echo ""
        echo "SOLUTION:"
        echo "  1. In OBS: Tools → Stop Virtual Camera"
        echo "  2. Stop Faceboard Nova (if running)"
        echo "  3. Run this script again"
        exit 1
    fi
    sleep 2
    echo "  ✓ Module unloaded"
fi

# Load module with ALL devices: video9 (for OBS) + video10-13 (for Faceboard Nova)
# OBS will automatically use video9 (first device), Faceboard Nova will use video10+
echo "  Loading v4l2loopback with 5 devices..."
echo "  - /dev/video9: Reserved for OBS Virtual Camera (first device)"
echo "  - /dev/video10-13: For Faceboard Nova"
sudo modprobe v4l2loopback devices=5 video_nr=9,10,11,12,13 card_label="cumcam","VCam-1","VCam-2","VCam-3","VCam-4" exclusive_caps=1

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully loaded module!"
    echo ""
    echo "Verifying devices were created:"
    all_ok=1
    for i in 9 10 11 12 13; do
        if [ -e "/dev/video$i" ]; then
            if v4l2-ctl --device /dev/video$i --info 2>/dev/null | grep -q "v4l2loopback"; then
                echo "  ✓ /dev/video$i exists (v4l2loopback)"
            else
                echo "  ⚠ /dev/video$i exists but not v4l2loopback"
            fi
        else
            echo "  ✗ /dev/video$i MISSING"
            all_ok=0
        fi
    done
    
    if [ $all_ok -eq 1 ]; then
        echo ""
        echo "✓ All devices created successfully!"
        echo ""
        echo "Device assignment:"
        echo "  - /dev/video9: OBS Virtual Camera (labeled 'cumcam', OBS will auto-select this)"
        echo "  - /dev/video10-13: Faceboard Nova (will use video10 or higher)"
        echo ""
        echo "Available devices:"
        v4l2-ctl --list-devices 2>/dev/null | grep -A 1 "VCam\|v4l2loopback\|cumcam"
        echo ""
        echo "⚠ IMPORTANT:"
        echo "  1. Restart Faceboard Nova for it to detect the new devices"
        echo "  2. OBS Virtual Camera will automatically use /dev/video9"
        echo "  3. Both can now run simultaneously without conflicts!"
    else
        echo ""
        echo "⚠ Some devices failed to create. Check dmesg for errors."
        exit 1
    fi
else
    echo ""
    echo "⚠ Failed to load module with new parameters."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check dmesg: dmesg | tail -20"
    echo "  2. Verify permissions: sudo usermod -a -G video $USER"
    exit 1
fi
