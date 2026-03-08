import os
paths_to_check = [
    '/mnt/f/datasets/nuscenes_lowlight',
    '/mnt/f/datasets/nuscenes_lowlight/v1.0-mini',
    '/mnt/f/datasets/nuscenes_lowlight/samples',
    '/mnt/f/datasets/nuscenes_lowlight/nuscenes_infos_temporal_val.pkl',
    '/mnt/f/datasets/nuscenes_lowlight/can_bus',
    '/mnt/f/datasets/nuscenes',
    '/mnt/f/datasets/nuscenes/v1.0-mini',
    '/mnt/f/datasets/nuscenes/samples',
    '/mnt/f/datasets/nuscenes/nuscenes_infos_temporal_val.pkl',
    '/mnt/f/datasets/nuscenes/can_bus',
]
for p in paths_to_check:
    exists = os.path.exists(p)
    is_dir = os.path.isdir(p) if exists else False
    print(f"{'OK' if exists else 'MISSING':>7} {'DIR' if is_dir else 'FILE' if exists else '   '} {p}")
    if exists and is_dir:
        items = os.listdir(p)[:10]
        for item in sorted(items):
            print(f"          -> {item}")
