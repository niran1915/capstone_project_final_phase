---
title: CrackVision non-crack segmentation
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: 3.10
---

# CrackVision non-crack ZeroGPU Space

The project Space is deployed privately at
`niranjan2005/crackvision-noncrack-gpu`, with the detector and FastSAM weights
uploaded. The root `start_windows.bat` points the local backend at this Space.
The account must remain logged in locally with `hf auth login` so the backend
can call the private Gradio endpoint.

This Space performs detector-guided FastSAM masks for concrete spalling, rebar,
and fissures. It deliberately excludes the crack class; local CrackVision keeps
using CRACKTO-SFNet V4 for cracks. The local app can call this Space for both
multi-defect segmentation and multi-class damage mapping.

## Deploy under your Hugging Face account

1. Create a new **Gradio Space** and select **ZeroGPU** hardware.
2. Upload `app.py` and `requirements.txt` from this folder.
3. Upload `backend/app/models/detector.pt` and `backend/app/models/FastSAM-s.pt`
   into the Space root as `detector.pt` and `FastSAM-s.pt`. Keep the Space private
   if you do not want to publish these weights.
4. Once the Space is running, copy its `owner/space-name` identifier.
5. In the local backend environment, set `CRACKVISION_GPU_SPACE=owner/space-name`
   and log in with `hf auth login` (the private Space client uses that saved token).
6. Install `gradio_client` in the backend environment and restart the backend.

ZeroGPU has queues and user quotas; it is suitable for intermittent use, not
guaranteed always-on inference. With no `CRACKVISION_GPU_SPACE` value, the app
continues using its existing local detector + FastSAM path.
