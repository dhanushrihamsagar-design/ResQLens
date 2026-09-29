# 🚨 ResQLens — Offline AI Disaster Vision & Hazard Prioritization

ResQLens is an offline AI-powered disaster vision system that analyzes disaster-scene images, detects relevant objects, and prioritizes potential hazards using a lightweight risk-reasoning layer.

The system combines **Qualcomm AI Hub's YOLOv8-N object detection model** with a custom hazard-prioritization engine to transform visual detections into understandable safety information.

---

## 🎯 Problem

During disasters such as floods and other emergencies, visual information can contain multiple objects and potential hazards. Quickly identifying relevant objects and interpreting their combination can help provide useful situational information.

Traditional object detection only identifies objects. It does not directly determine whether a scene represents a potential hazard.

ResQLens addresses this by adding a separate reasoning layer after object detection.

---

## 💡 Solution

ResQLens follows this pipeline:

```text
Disaster Image
      ↓
Image Preprocessing
      ↓
Qualcomm AI Hub YOLOv8-N
      ↓s
Object Detection
      ↓
Confidence Filtering
      ↓
Class-Aware Non-Maximum Suppression
      ↓
Risk Reasoning Engine
      ↓
Potential Hazard Level
      ↓
Explanation + Recommended Safety Action