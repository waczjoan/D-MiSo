#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games.dynamic.pcd_splatting.scene.pcd_gaussian_model import PcdGaussianModel


def write_simple_obj(pseudomesh, filepath, verbose=False):

    faces = torch.range(
        0, pseudomesh.shape[0] * 3 - 1
    ).reshape(pseudomesh.shape[0], 3)

    vertice = pseudomesh.reshape(pseudomesh.shape[0] * 3, 3)

    mesh_v = vertice.detach().cpu().numpy()
    mesh_f = faces
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('pseudomesh saved to: ', filepath)


def save_pseudomesh(model_path, name, iteration, gaussians, deform):
    pseudomesh_path = os.path.join(model_path, name, "ours_{}".format(iteration), "core_triangle_soup")
    attached_pseudomesh_path = os.path.join(model_path, name, "ours_{}".format(iteration), "sub_triangle_soup")

    makedirs(pseudomesh_path, exist_ok=True)
    makedirs(attached_pseudomesh_path, exist_ok=True)

    # Example fids, please change it if you need it:
    fids = torch.range(0, 200).float() * 0.005

    for idx, fid in enumerate(tqdm(fids, desc="Pseudomesh saving progress")):
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1).cuda()
        d_v1, d_v2, d_v3, d_rot = deform.step(
            gaussians.pseudomesh[:, 0].detach(),
            gaussians.pseudomesh[:, 1].detach(),
            gaussians.pseudomesh[:, 2].detach(),
            time_input
        )

        v1, v2, v3 = gaussians.calc_vertices(d_v1, d_v2, d_v3, d_rot)

        scales, rotations = gaussians._prepare_scaling_rot(v1, v2, v3)
        s0 = torch.ones(scales.shape[0], 1).cuda() * gaussians.eps_s0
        scales = torch.cat([s0, gaussians.scaling_activation(scales[:, [-2, -1]])], dim=1)
        rotations = gaussians.rotation_activation(rotations)
        means3D = gaussians.get_xyz + d_v1

        pseudomesh = gaussians.create_faces(means3D, scales, rotations)

        if gaussians.use_attached_gauss:
            scales = gaussians.get_attached_scales(scales)
            rotations = gaussians.get_attached_rotations(rotations, time_input)
            means3D = gaussians.calc_attached_gauss(v1, v2, v3)

        attached_pseudomesh = gaussians.create_faces(means3D, scales, rotations)

        write_simple_obj(
            pseudomesh=pseudomesh,
            filepath=os.path.join(pseudomesh_path, f"core_triangle_soup_time_{'{0:.4f}'.format(fid)}" + ".obj")
        )

        write_simple_obj(
            pseudomesh=attached_pseudomesh,
            filepath=os.path.join(attached_pseudomesh_path, f"sub_triangle_soup_time_{'{0:.4f}'.format(fid)}" + ".obj")
        )


def save_pseudomeshes(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians = PcdGaussianModel(dataset.sh_degree, dataset.deform_width, dataset.deform_depth, dataset.is_blender, dataset.is_6dof)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = gaussians.deform_model
        deform.load_weights(dataset.model_path, iteration)
        gaussians.load_time_weights(dataset.model_path, iteration)

        render_func = save_pseudomesh

        render_func(
            dataset.model_path,"triangle_soups", scene.loaded_iter,
            gaussians, deform
        )



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default='best')
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    save_pseudomeshes(model.extract(args), args.iteration, pipeline.extract(args))
