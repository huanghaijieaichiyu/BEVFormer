#!/bin/bash
echo "=== Checking /mnt/f/datasets/nuscenes_lowlight ==="
ls /mnt/f/datasets/nuscenes_lowlight/ 2>&1 | head -20
echo ""
echo "=== Checking v1.0-mini in lowlight ==="
ls /mnt/f/datasets/nuscenes_lowlight/v1.0-mini/ 2>&1 | head -5
echo ""
echo "=== Checking pkl in lowlight ==="
ls -la /mnt/f/datasets/nuscenes_lowlight/nuscenes_infos_temporal_val.pkl 2>&1
ls -la /mnt/f/datasets/nuscenes_lowlight/can_bus/ 2>&1 | head -5
echo ""
echo "=== Checking /mnt/f/datasets/nuscenes ==="
ls /mnt/f/datasets/nuscenes/ 2>&1 | head -20
echo ""
echo "=== Checking v1.0-mini in normal ==="
ls /mnt/f/datasets/nuscenes/v1.0-mini/ 2>&1 | head -5
echo ""
echo "=== Checking pkl in normal ==="
ls -la /mnt/f/datasets/nuscenes/nuscenes_infos_temporal_val.pkl 2>&1
ls -la /mnt/f/datasets/nuscenes/can_bus/ 2>&1 | head -5
echo ""
echo "=== Checking samples in lowlight ==="
ls /mnt/f/datasets/nuscenes_lowlight/samples/ 2>&1 | head -10
echo ""
echo "=== Checking samples in normal ==="
ls /mnt/f/datasets/nuscenes/samples/ 2>&1 | head -10
