# QUICK FIX - Virtual Camera Conflict

## The Problem
- OBS Virtual Camera uses `/dev/video9` (first v4l2loopback device)
- Faceboard Nova needs separate devices
- Both conflict when trying to use same device

## The Solution (Run These Commands)

**Step 1: Stop OBS Virtual Camera**
- In OBS: Click "Stop Virtual Camera" button
- Or: Tools → Stop Virtual Camera

**Step 2: Run these commands in your terminal:**

```bash
# Unload existing module
sudo modprobe -r v4l2loopback

# Load with 5 devices: video9 (OBS) + video10-13 (Faceboard Nova)
sudo modprobe v4l2loopback devices=5 video_nr=9,10,11,12,13 card_label="OBS-VCam","VCam-1","VCam-2","VCam-3","VCam-4" exclusive_caps=1

# Verify devices were created
ls -la /dev/video{9..13}
```

**Step 3: Verify it worked:**
```bash
v4l2-ctl --list-devices | grep -A 1 "v4l2loopback"
```

You should see 5 devices listed.

**Step 4:**
- Restart Faceboard Nova (it will use video10+)
- Start OBS Virtual Camera (it will use video9)
- Both work simultaneously!

## Why This Works

Research shows:
- OBS **always** uses the **first** v4l2loopback device it finds
- OBS **cannot** be configured to use a different device
- Solution: Load video9 **first** (OBS gets it), then video10+ (our app uses these)
