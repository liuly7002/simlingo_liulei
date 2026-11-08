# -*- coding: utf-8 -*-
"""Optional read-only live trajectory visualization for Bench2Drive evaluation.

Plotting uses predicted ego-local (x,y) coordinates (metres). No map geometry
or legal road-line inference is implied by the BEV grid. The camera projection
uses SimLingo's existing project_points helper as an approximate overlay.
"""
import os
from pathlib import Path

import cv2
import numpy as np

from team_code.simlingo_utils import get_camera_intrinsics, project_points


class TrajectoryVisualizer:
    WINDOW = 'SimLingo Stage-2 | camera + ego-local BEV'
    COLORS = {
        'Route (20)': (235, 150, 30),    # BGR: blue
        'Initial (10)': (0, 215, 255),  # BGR: yellow
        'Final (10)': (40, 220, 40),    # BGR: green
    }

    def __init__(self, show=False, save=False, every=5, save_dir=None):
        self.show = bool(show)
        self.save = bool(save)
        self.every = max(1, int(every))
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.initialized = False
        if self.save and self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.show and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            print('[TrajectoryViz] No desktop DISPLAY/WAYLAND_DISPLAY. Saving only.', flush=True)
            self.show = False

    @staticmethod
    def as_points(value):
        if value is None:
            return None
        if hasattr(value, 'detach'):
            value = value.detach().float().cpu().numpy()
        points = np.asarray(value, dtype=np.float32)
        if points.ndim == 3:
            points = points[0]
        if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
            return None
        return points[:, :2].copy()

    @staticmethod
    def _put(frame, text, origin, color=(245, 245, 245), size=0.56):
        cv2.putText(frame, str(text), origin, cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, 1, cv2.LINE_AA)

    def draw_camera(self, image, trajectories, height=520, width=880):
        # Agent's front camera comes from CARLA as BGR; do not change agent input.
        front = np.asarray(image).copy()
        if front.ndim != 3 or front.shape[2] < 3:
            raise ValueError('Expected front camera image [H,W,3+]')
        front = front[:, :, :3]
        h, w = front.shape[:2]
        K = np.asarray(get_camera_intrinsics(w, h, 110), dtype=np.float64)
        for label, pts in trajectories.items():
            if pts is None:
                continue
            color = self.COLORS[label]
            # Preserve SimLingo's original approximate 2-D camera projection.
            projected = project_points(pts, K)
            for point in projected:
                u, v = np.asarray(point).ravel()[:2]
                if np.isfinite(u) and np.isfinite(v) and 0 <= u < w and 0 <= v < h:
                    cv2.circle(front, (int(round(u)), int(round(v))), 5, color, -1, cv2.LINE_AA)
        out = cv2.resize(front, (width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(out, (0, 0), (width, 35), (25, 25, 25), -1)
        self._put(out, 'Front camera | projected waypoints (approx.)', (12, 24))
        return out

    def draw_bev(self, trajectories, side=520):
        bev = np.full((side, side, 3), (34, 38, 44), dtype=np.uint8)
        # Grid in metres. Ego x = forward; ego y = lateral.
        # Positive ego-y appears on the left of this visualization.
        min_x, max_x, min_y, max_y = -5., 35., -15., 15.
        pad = 42
        sx = (side-2*pad)/(max_y-min_y)
        sy = (side-2*pad)/(max_x-min_x)
        def pix(x, y):
            u = round(pad + (max_y-float(y))*sx)
            v = round(side-pad - (float(x)-min_x)*sy)
            return int(u), int(v)
        for x in range(0, 36, 5):
            cv2.line(bev, pix(x, min_y), pix(x, max_y), (65, 69, 76), 1)
            self._put(bev, str(x), (4, max(12, min(side-10, pix(x, 0)[1]+4))), size=.38)
        for y in range(-15, 16, 5):
            cv2.line(bev, pix(min_x, y), pix(max_x, y), (65, 69, 76), 1)
            self._put(bev, str(y), (pix(0,y)[0]-6, side-12), size=.38)
        cv2.line(bev, pix(min_x, 0), pix(max_x, 0), (110, 115, 120), 1)
        cv2.circle(bev, pix(0, 0), 8, (255, 255, 255), -1)
        cv2.line(bev, pix(0, 0), pix(2, 0), (255,255,255), 2)
        for label, pts in trajectories.items():
            if pts is None:
                continue
            color = self.COLORS[label]
            xy = [pix(p[0], p[1]) for p in pts if np.isfinite(p).all()]
            if len(xy)>1:
                cv2.polylines(bev, [np.asarray(xy, np.int32)], False, color, 2, cv2.LINE_AA)
            for k, xy_pt in enumerate(xy):
                cv2.circle(bev, xy_pt, 4 if k != len(xy)-1 else 6, color, -1, cv2.LINE_AA)
        cv2.rectangle(bev, (0, 0), (side, 110), (25,25,25), -1)
        self._put(bev, 'Ego-local BEV | x forward / y lateral (metres)', (10,22), size=.48)
        for i, label in enumerate(trajectories):
            y = 47+i*19
            cv2.line(bev, (12,y-5), (37,y-5), self.COLORS[label], 3)
            self._put(bev, label, (45,y), size=.50)
        return bev

    def update(self, frame, camera, route, initial, final, velocity=None, answer=None):
        if not (self.show or self.save) or frame % self.every:
            return
        trajectories = {
            'Route (20)': self.as_points(route),
            'Initial (10)': self.as_points(initial),
            'Final (10)': self.as_points(final),
        }
        image = self.draw_camera(camera, trajectories)
        bev = self.draw_bev(trajectories)
        canvas = np.concatenate((image, bev), axis=1)
        foot = np.full((56, canvas.shape[1], 3), (25,25,25), dtype=np.uint8)
        header = 'frame={}{}'.format(frame, '' if velocity is None else '   speed={:.2f} m/s'.format(float(velocity)))
        self._put(foot, header, (12, 20))
        if answer:
            short = ' '.join(str(answer).split())[:170]
            self._put(foot, 'Language: '+short, (12, 43), size=.45)
        canvas = np.concatenate((canvas, foot), axis=0)
        if self.save and self.save_dir is not None:
            cv2.imwrite(str(self.save_dir / ('frame_%06d.png' % frame)), canvas)
            np.savez_compressed(str(self.save_dir / ('frame_%06d.npz' % frame)),
                                route=trajectories['Route (20)'] if trajectories['Route (20)'] is not None else np.zeros((0,2)),
                                initial=trajectories['Initial (10)'] if trajectories['Initial (10)'] is not None else np.zeros((0,2)),
                                final=trajectories['Final (10)'] if trajectories['Final (10)'] is not None else np.zeros((0,2)))
        if self.show:
            if not self.initialized:
                cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
                self.initialized = True
            cv2.imshow(self.WINDOW, canvas)
            cv2.waitKey(1)

    def close(self):
        if self.initialized:
            try:
                cv2.destroyWindow(self.WINDOW)
            except cv2.error:
                pass
        self.initialized = False
