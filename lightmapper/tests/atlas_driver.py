"""Runs inside Blender: runs bake.main on a job once per entry of driver.json (each with its own overrides), spying on every
bake call, and writes atlas.json (what each bake saw and what bake.log said) plus each run's out npz and topped-up masks."""

import importlib.util
import json
import os
import sys
import time

import bpy
import numpy as np


def reaches(start):
    # every node type on a path into the start node's inputs
    seen, stack, types = set(), [start], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        types.add(node.type)
        for socket in node.inputs:
            for link in socket.links:
                stack.append(link.from_node)
    return types


def describe(material):
    tree = material.node_tree
    output = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL')
    types = sorted(reaches(output))
    colours = {}
    for node in tree.nodes:
        if node.type in ('BSDF_DIFFUSE', 'BSDF_TRANSLUCENT') and node.type in types:
            socket = node.inputs['Color']
            colours[node.type] = sorted(reaches(socket.links[0].from_node)) if socket.links else list(socket.default_value)
    return {'types': types, 'colours': colours}


class ObjectOps:
    def __init__(self, real, calls, counts):
        self.real, self.calls, self.counts = real, calls, counts

    def __getattr__(self, name):
        return getattr(self.real.ops.object, name)

    def bake(self, **kwargs):
        real = self.real
        context = real.context
        if 'before' not in self.counts:
            self.counts['before'] = counts()
        chosen = list(context.selected_objects)
        self.calls.append({
            'counts': counts(), 'type': kwargs.get('type'), 'samples': context.scene.cycles.samples, 'seed': context.scene.cycles.seed,
            'selected': [o.name for o in chosen],
            'objects': {o.name: {'collections': [c.name for c in real.data.collections if o.name in c.objects],
                                 'flagged': bool(o.get('flagged', False)), 'triangles': len(o.data.polygons),
                                 'visible': {k: bool(getattr(o, k)) for k in
                                             ('visible_camera', 'visible_diffuse', 'visible_glossy', 'visible_shadow',
                                              'visible_transmission', 'visible_volume_scatter')},
                                 'materials': [describe(m) for m in o.data.materials]} for o in chosen},
            'images': {i.name: list(i.size) for i in real.data.images
                       if i.name == 'atlas' or i.name.startswith(('page_', 'mask_'))}})
        return real.ops.object.bake(**kwargs)


class Ops:
    def __init__(self, real, calls, counts):
        self.real, self.object = real, ObjectOps(real, calls, counts)

    def __getattr__(self, name):
        return getattr(self.real.ops, name)


class Spy:
    def __init__(self, real, calls, counts):
        self.real, self.ops = real, Ops(real, calls, counts)

    def __getattr__(self, name):
        return getattr(self.real, name)


def counts():
    return {'objects': len(bpy.data.objects), 'meshes': len(bpy.data.meshes), 'materials': len(bpy.data.materials)}


def main():
    root = sys.argv[sys.argv.index('--') + 1]
    spec = importlib.util.spec_from_file_location('lm_bake', os.path.join(os.environ['LIGHTMAPPER_BAKE_DIR'], 'bake.py'))
    bake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bake)
    with open(os.path.join(root, 'driver.json')) as handle:
        runs = json.load(handle)['runs']
    with open(os.path.join(root, 'job.json')) as handle:
        base = json.load(handle)
    real_bpy, real_log, real_plan = bake.bpy, bake.log, bake.plan_extra_passes
    real_owner, real_scatter, real_place = bake.subtexel_owner, bake.scatter_atlas, bake.place_atlas
    report = {}
    for run in runs:
        name = run['name']
        job = {**base, **run.get('job', {}), 'out': f'out_{name}.npz'}
        path = os.path.join(root, f'job_{name}.json')
        with open(path, 'w') as handle:
            json.dump(job, handle)
        calls, lines, masks, tally, roles = [], [], [], {}, []

        def log(message, lines=lines):
            lines.append(message)
            real_log(message)

        def plan(owned, base_spp, target, masks=masks):
            tiers = real_plan(owned, base_spp, target)
            union = np.zeros(owned.shape, bool)
            for _, mask in tiers:
                union |= mask
            masks.append(union[::-1].copy())
            return tiers

        def no_owner(*args):
            # (every written sub-texel unowned, so every tier must be baked through the page mask)
            owner = real_owner(*args)
            owner[:] = -1
            return owner

        def placed(*args):
            plan = real_place(*args)
            if plan is not None:
                roles.append(np.bincount(plan.role, minlength=3).tolist())
            return plan

        def nothing_written(atlas, values, shape):
            buffer, _ = real_scatter(atlas, values, shape)
            buffer.fill(-1)
            return buffer, len(atlas.slots)

        bake.bpy, bake.log, bake.plan_extra_passes = Spy(real_bpy, calls, tally), log, plan
        bake.place_atlas = placed
        if run.get('unowned'):
            bake.subtexel_owner = no_owner
        if run.get('unwritten'):
            bake.scatter_atlas = nothing_written
        sys.argv = [sys.argv[0], '--', path]
        started = time.perf_counter()
        try:
            bake.main()
        finally:
            bake.bpy, bake.log, bake.plan_extra_passes = real_bpy, real_log, real_plan
            bake.subtexel_owner, bake.scatter_atlas, bake.place_atlas = real_owner, real_scatter, real_place
        np.savez(os.path.join(root, f'masks_{name}.npz'), **{f'mask_{k}': m for k, m in enumerate(masks)})
        report[name] = {'calls': calls, 'log': lines, 'before': tally.get('before'), 'after': counts(), 'roles': roles,
                        'seconds': time.perf_counter() - started}
    with open(os.path.join(root, 'atlas.json'), 'w') as handle:
        json.dump(report, handle)


main()
