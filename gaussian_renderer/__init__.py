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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_v1, d_v2, d_v3, d_rot, is_6dof=False,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_v1) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_v1, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_v1

    opacity = pc.get_opacity
    opacity = torch.cat((opacity, pc.get_mini_opacity))

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        v1 = pc.pseudomesh[:, 0]
        v2 = pc.pseudomesh[:, 1]
        v3 = pc.pseudomesh[:, 2]

        _v2 = v2 - v1
        _v3 = v3 - v1

        v1 = v1 + d_v1
        v2 = v2 + d_v1 + _v2/torch.linalg.vector_norm(_v2, dim=-1, keepdim=True) * d_v2 + d_rot
        v3 = v3 + d_v1 + _v3/torch.linalg.vector_norm(_v3, dim=-1, keepdim=True) * d_v3 + d_rot

        scales, rotations = pc._prepare_scaling_rot(v1, v2, v3)
        s0 = torch.ones(scales.shape[0], 1).cuda() * pc.eps_s0
        scales = torch.cat([s0, pc.scaling_activation(scales[:, [-2, -1]])], dim=1)
        scales = torch.cat((scales, pc.get_mini_scales(scales)))
        rotations = pc.rotation_activation(rotations)
        rotations = torch.cat((rotations, pc.get_mini_rotations(rotations)))
    
        means3D = torch.cat((means3D, pc.calc_mini_gauss(v1, v2, v3)))
        screenspace_points = torch.zeros_like(means3D, requires_grad=True, device="cuda")
        means2D = screenspace_points

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            shs = torch.cat((shs, pc.get_mini_shs))
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}
